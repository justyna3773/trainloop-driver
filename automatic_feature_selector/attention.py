"""
attention_clean.py
------------------
Kod ekstraktora uwagi (wariant content-based, AV + wspólny embedding + P_idx).
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces


class AttentionExtractor(BaseFeaturesExtractor):
    """
    Ekstraktor cech oparty na mechanizmie dot-product self-attention
    nad metrykami stanu, traktowanymi jako "tokeny" (features-as-tokens).

    Główne cechy:
    - wspólny embedding skalarnej metryki: Linear(1 -> d_embed),
    - uczone wektorowe kodowania indeksu metryki P_idx (feature / positional encodings),
    - klasyczne Q/K/V zależne od embeddingów (content-based attention),
    - multi-head attention z agregacją głów (globalne wagi głów lub mean/sum/max),
    - wartości V pochodzą z embeddingów E, a wyjście to A V,
    - pooling po metrykach -> wektor w przestrzeni embeddingu, opcjonalnie zprojekowany
      do final_out_dim.

    Argumenty konstruktora (__init__):
        observation_space: Przestrzeń obserwacji `gym.spaces.Box`; ostatni wymiar to liczba cech (N).
        d_embed: Rozmiar wektorów osadzeń (embedding) dla każdej metryki.
        d_k: Rozmiar wektora zapytań/kluczy na głowę (head) uwagi.
        n_heads: Liczba głowic uwagi.
        attn_temp: Temperatura w normalizacji logitów uwagi (większa → „łagodniejszy” softmax).
        head_agg: Agregacja głowic: "mean" | "sum" | "max".
        use_posenc: Czy dodawać uczone osadzenia pozycyjne cech (stałe P_idx).
        final_out_dim: Jeśli podane, rzutuje wyjście do tej liczby wymiarów (Linear + opcjonalna aktywacja).
        out_layernorm: Czy stosować LayerNorm na wyjściu (gdy final_out_dim nie jest ustawione).
        out_activation: Aktywacja na wyjściu: "tanh" | "relu" | None.
        use_residual: Czy dodać połączenie rezydualne: E + w * Attention(E).
        residual_weight: Współczynnik w połączeniu rezydualnym.
        freeze: Jeśli True — zamraża wszystkie parametry ekstraktora (bez uczenia).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        d_embed: int = 32,
        d_k: int = 16,
        n_heads: int = 1,
        attn_temp: float = 1.0,
        head_agg: str = "mean",
        use_posenc: bool = True,
        # stare parametry zostawiam w sygnaturze dla kompatybilności, ale ignoruję:
        use_content_embed: bool = False,
        alpha_mode: str = "global",
        alpha_init: float = 0.5,
        learn_alpha: bool = True,
        alpha_mlp_hidden: int = 32,
        alpha_pool: str = "mean",
        final_out_dim: Optional[int] = None,
        out_layernorm: bool = False,
        out_activation: Optional[str] = "tanh",
        use_residual: bool = True,
        residual_weight: float = 1.0,
        freeze: bool = False,
    ):
        n_metrics = int(observation_space.shape[-1])
        self.n_metrics = n_metrics
        self.d_embed = int(d_embed)
        self.d_k = int(d_k)
        self.n_heads = int(n_heads)
        self.head_agg = head_agg.lower()
        self.mode = "content_only"
        self.attn_norm = "row_softmax"
        self.attn_temp = float(attn_temp)
        self.use_posenc = bool(use_posenc)
        self.use_residual = bool(use_residual)
        self.residual_weight = float(residual_weight)

        assert self.n_heads >= 1
        assert self.head_agg in {"mean", "sum", "max"}

        # >>> WYMIAR WYJŚCIA EKSTRAKTORA <<<
        # Po attention i poolingu po metrykach mamy wektor [B, d_embed].
        raw_dim = self.d_embed
        features_dim = int(final_out_dim) if final_out_dim is not None else raw_dim
        super().__init__(observation_space, features_dim=features_dim)

        # Shared embedding: Linear(1 -> d_embed) współdzielony przez wszystkie metryki
        self.value_embed = nn.Linear(1, self.d_embed)

        # LayerNorm na embeddingach
        self.ln_e = nn.LayerNorm(self.d_embed)

        # Uczone "positional encodings" po osi metryk (identyfikatory metryk)
        if self.use_posenc:
            self.P_idx = nn.Parameter(
                th.randn(n_metrics, self.d_embed) * 0.02
            )  # [N, d_embed]
        else:
            self.P_idx = None

        # Stare pola zostawione dla kompatybilności (ale nie są używane):
        self.use_content_embed = False
        self.embedders = None
        self.register_buffer("value_projection", th.ones(1, self.d_embed))

        # --- Q/K/V: CZYSTO CONTENT-BASED (Z E, KTÓRE ZAWIERA P_idx) ---

        # Q, K: projekcje embeddingów na n_heads * d_k
        self.Wq = nn.Linear(self.d_embed, self.n_heads * self.d_k, bias=False)
        self.Wk = nn.Linear(self.d_embed, self.n_heads * self.d_k, bias=False)

        # V: projekcja embeddingów na d_embed (wspólna dla wszystkich głów)
        self.Wv = nn.Linear(self.d_embed, self.d_embed, bias=False)

        # --- Multi-head: globalnie uczone wagi głów (jeśli H > 1) ---

        if self.n_heads > 1:
            self.head_logits = nn.Parameter(th.zeros(self.n_heads))
        else:
            self.head_logits = None

        # --- Warstwy wyjściowe ---

        self.out_layernorm = bool(out_layernorm)
        self.out_activation = out_activation.lower() if out_activation is not None else None
        if features_dim != raw_dim:
            layers = [nn.LayerNorm(raw_dim), nn.Linear(raw_dim, features_dim)]
            if self.out_activation == "tanh":
                layers.append(nn.Tanh())
            elif self.out_activation == "relu":
                layers.append(nn.ReLU())
            self.post_proj = nn.Sequential(*layers)
            self.out_ln = None
            self.out_act = None
        else:
            self.post_proj = None
            self.out_ln = nn.LayerNorm(raw_dim) if self.out_layernorm else None
            if self.out_activation == "tanh":
                self.out_act = nn.Tanh()
            elif self.out_activation == "relu":
                self.out_act = nn.ReLU()
            else:
                self.out_act = None

        # --- Diagnostyka ---

        self.attn_matrix: Optional[th.Tensor] = None
        self.metric_importance: Optional[th.Tensor] = None
        self.contrib_importance: Optional[th.Tensor] = None
        self.attn_history = deque(maxlen=1000)
        self.contrib_history = deque(maxlen=1000)
        self.total_steps = 0
        self.register_buffer("active_mask", th.ones(self.n_metrics, dtype=th.float32))

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    # ---------------- EMBEDDING ----------------

    def _embed(self, x: th.Tensor) -> th.Tensor:
        """
        Wspólny embedding cech:
        - Linear(1 -> d_embed) współdzielony między metrykami,
        - dodanie wektorów P_idx (identyfikatory metryk),
        - LayerNorm.

        Wejście: x [B, N]
        Wyjście: E [B, N, d_embed]
        """
        if hasattr(self, "active_mask"):
            x = x * self.active_mask.to(x.device).view(1, -1)

        B, N = x.shape
        assert N == self.n_metrics, f"Spodziewano się {self.n_metrics} metryk, otrzymano {N}"

        # Shared token embedding
        E = self.value_embed(x.unsqueeze(-1))  # [B, N, d_embed]

        # Identyfikatory metryk (positional encodings po osi features)
        if self.P_idx is not None:
            E = E + self.P_idx.view(1, self.n_metrics, self.d_embed)

        # Normalizacja
        E = self.ln_e(E)

        return E

    # ---------------- ATTENTION ----------------

    def _attention_logits(self, E: th.Tensor) -> th.Tensor:
        """
        Oblicza logity uwagi per głowa na bazie embeddingów E (content-based).

        Zwraca:
            scores: [B, H, N, N]
        """
        B, N, _ = E.shape

        # Q, K: [B, N, H * d_k] -> [B, H, N, d_k]
        q = self.Wq(E).view(B, N, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        k = self.Wk(E).view(B, N, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # scores: [B, H, N, N]
        scores = th.matmul(q, k.transpose(-2, -1)) / (float(self.d_k) ** 0.5)

        # temperatura uwagi
        scores = scores / max(1e-6, self.attn_temp)
        return scores

    def _normalize_heads(self, scores: th.Tensor) -> th.Tensor:
        # row_softmax po ostatnim wymiarze
        return F.softmax(scores, dim=-1)

    def _aggregate_heads(self, heads: th.Tensor) -> th.Tensor:
        """
        Agreguje wiele głów uwagi w jedną macierz uwagi.

        heads: [B, H, N, N]
        Zwraca:
            A: [B, N, N]
        """
        B, H, N, _ = heads.shape

        # jedna głowa -> nic nie mieszamy
        if H == 1:
            return heads[:, 0]

        # globalnie uczone wagi głów
        # if self.head_logits is not None:
        #     weights = F.softmax(self.head_logits, dim=0)  # [H]
        #     A = (heads * weights.view(1, H, 1, 1)).sum(dim=1)  # [B, N, N]
        #     return A

        # fallback: mean/sum/max
        if self.head_agg == "mean":
            A = heads.mean(dim=1)
        elif self.head_agg == "sum":
            A = heads.sum(dim=1)
        else:  # "max"
            A, _ = heads.max(dim=1)

        if self.head_agg in {"sum", "max"}:
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return A

    # ---------------- DIAGNOSTYKA ----------------

    def _diagnostics(self, A: th.Tensor, x: th.Tensor) -> None:
        with th.no_grad():
            # A: [B, N, N]
            vec = A.mean(dim=1)  # [B, N]
            metric = vec / vec.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            contrib = A.abs().sum(dim=1) * x.abs()
            contrib = contrib / contrib.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            self.metric_importance = metric.detach()
            self.contrib_importance = contrib.detach()
            self.attn_history.append(self.metric_importance.mean(dim=0).cpu())
            self.contrib_history.append(self.contrib_importance.mean(dim=0).cpu())
            self.total_steps += 1

    # ---------------- FORWARD ----------------

    def _forward_batch(self, x: th.Tensor) -> th.Tensor:
        """
        Główna funkcja forward dla batcha obserwacji.

        Wejście: x [B, N]
        Wyjście: out [B, features_dim]
        """
        # Krok 1: embedding cech
        E = self._embed(x)  # [B, N, d_embed]

        # Krok 2: logity uwagi + normalizacja
        scores = self._attention_logits(E)          # [B, H, N, N]
        heads = self._normalize_heads(scores)       # [B, H, N, N]

        # Krok 3: agregacja głów -> jedna macierz uwagi A
        A = self._aggregate_heads(heads)            # [B, N, N]

        # Zapis macierzy uwagi
        self.attn_matrix = A.detach()

        # Krok 4: Values z embeddingów i klasyczne A V
        V = self.Wv(E)                              # [B, N, d_embed]
        mixed_embed = th.matmul(A, V)               # [B, N, d_embed]

        # Krok 5: połączenie rezydualne w przestrzeni embeddingów
        if self.use_residual:
            E_out = E + self.residual_weight * mixed_embed
        else:
            E_out = mixed_embed

        # Krok 6: diagnostyka (na bazie A i oryginalnych skalarów x)
        self._diagnostics(A, x)

        # Krok 7: pooling po metrykach -> wektor stanu [B, d_embed]
        pooled = E_out.mean(dim=1)                  # [B, d_embed]

        # Krok 8: post-processing do features_dim
        if self.post_proj is not None:
            out = self.post_proj(pooled)
        else:
            out = pooled
            if self.out_ln is not None:
                out = self.out_ln(out)
            if self.out_act is not None:
                out = self.out_act(out)
        return out

    def forward(self, obs: th.Tensor) -> th.Tensor:
        if obs.dim() == 2:
            return self._forward_batch(obs)
        if obs.dim() == 3:
            T, B, N = obs.shape
            out = self._forward_batch(obs.view(T * B, N))
            return out.view(T, B, -1)
        raise ValueError(f"Nieobsługiwany kształt obserwacji: {tuple(obs.shape)}")

    def set_attn_temperature(self, new_temp: float):
        self.attn_temp = float(new_temp)


# ----------------- TRENING Z SB3 -----------------

def train_model(env, args):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    policy_kwargs = dict(
        features_extractor_class=AttentionExtractor,
        features_extractor_kwargs=dict(
            d_embed=7,           # wymiar embeddingu / features_dim gdy final_out_dim=None
            d_k=4,
            n_heads=2,
            head_agg="mean",
            use_posenc=True,
            # parametry alpha_* i use_content_embed są ignorowane w implementacji,
            # ale można je zostawić dla kompatybilności:
            # alpha_mode="global",
            # alpha_init=0.5,
            # learn_alpha=False,
            # use_content_embed=False,
            attn_temp=1.2,
            final_out_dim=7,          # wyjście ekstraktora będzie miało wymiar 7
            out_layernorm=False,
            out_activation="tanh",
            use_residual=False,
        ),
        lstm_hidden_size=64,
        n_lstm_layers=1,
        net_arch=[dict(pi=[16, 16], vf=[16, 16])],
        activation_fn=th.nn.ReLU,
        ortho_init=True,
        normalize_images=False,
    )

    if args.policy == 'MlpLstmPolicy':
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            learning_rate=0.0003,
            vf_coef=1,
            clip_range_vf=10.0,
            max_grad_norm=1,
            gamma=0.95,
            ent_coef=0.001,
            clip_range=0.1,
            verbose=1,
            seed=int(args.seed),
            tensorboard_log='./output_malota/att_new_settings'
        )
    elif args.policy == 'MlpPolicy':
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(
                **policy_kwargs,
                optimizer_kwargs=dict(eps=1e-5),
            ),
            n_steps=2048,
            n_epochs=10,
            learning_rate=0.00003,
            vf_coef=1,
            clip_range_vf=10.0,
            max_grad_norm=1,
            gamma=0.95,
            ent_coef=0.001,
            clip_range=0.05,
            verbose=1,
            seed=int(args.seed),
            tensorboard_log="./output_malota/gawrl_mlp"
        )
    return model
