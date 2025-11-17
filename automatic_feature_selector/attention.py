"""
attention_clean.py
------------------
Kod ekstraktora uwagi.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces


"""
attention_clean.py
------------------
Kod ekstraktora uwagi.
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
    Uproszczony ekstraktor oparty na mechanizmie uwagi (wariant hybrydowy).

    Argumenty konstruktora (__init__):
        observation_space: Przestrzeń obserwacji `gym.spaces.Box`; ostatni wymiar to liczba cech (N).
        d_embed: Rozmiar wektorów osadzeń (embedding) dla każdej cechy.
        d_k: Rozmiar wektora zapytań/kluczy na głowę (head) uwagi.
        n_heads: Liczba głów uwagi.
        attn_temp: Temperatura w normalizacji logitów uwagi (większa → „łagodniejszy” softmax).
        head_agg: Agregacja głów: "mean" | "sum" | "max".
        use_posenc: Czy dodawać indeksowe osadzenia pozycyjne cech (stałe P_idx).
        use_content_embed: Czy używać osobnych liniowych embedderów dla wartości cech (zamiast skalarnego skalowania).
        alpha_mode: Sposób wyznaczania α łączącego część indeksową i zależną od treści: "global" lub "pool".
        alpha_init: Wartość początkowa α (w przypadku trybu "global").
        learn_alpha: Czy uczyć skalar α (dla trybu "global").
        alpha_mlp_hidden: Rozmiar warstwy ukrytej MLP dla α (dla trybu innego niż "global").
        alpha_pool: Rodzaj pooling'u wejścia przed MLP α: "mean" lub "max".
        final_out_dim: Jeśli podane, rzutuje wyjście do tej liczby wymiarów (Linear + opcjonalna aktywacja).
        out_layernorm: Czy stosować LayerNorm na wyjściu (gdy final_out_dim nie jest ustawione).
        out_activation: Aktywacja na wyjściu: "tanh" | "relu" | None.
        use_residual: Czy dodać połączenie rezydualne: x + w * Attention(x).
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
        self.mode = "generalized"  # Stały tryb
        self.attn_norm = "row_softmax"  # Stała normalizacja
        self.attn_temp = float(attn_temp)
        self.use_posenc = bool(use_posenc)
        self.use_content_embed = bool(use_content_embed)
        self.use_residual = bool(use_residual)
        self.residual_weight = float(residual_weight)

        assert self.n_heads >= 1
        assert self.head_agg in {"mean", "sum", "max"}

        raw_dim = n_metrics
        features_dim = int(final_out_dim) if final_out_dim is not None else raw_dim
        super().__init__(observation_space, features_dim=features_dim)

        if self.use_content_embed:
            self.embedders = nn.ModuleList([nn.Linear(1, self.d_embed) for _ in range(n_metrics)])
        else:
            self.embedders = None
            self.register_buffer("value_projection", th.ones(1, self.d_embed))
        #self.ln_e = nn.Identity()
        self.ln_e = nn.LayerNorm(self.d_embed)
        self.P_idx = nn.Parameter(th.randn(n_metrics, self.d_embed) * 0.02) if self.use_posenc else None

        self.Wq_c = nn.Linear(self.d_embed, self.n_heads * self.d_k, bias=False)
        self.Wk_c = nn.Linear(self.d_embed, self.n_heads * self.d_k, bias=False)
        self.Q_idx = nn.Parameter(th.randn(self.n_heads, n_metrics, self.d_k) * 0.02)
        self.K_idx = nn.Parameter(th.randn(self.n_heads, n_metrics, self.d_k) * 0.02)

        self.alpha_mode = alpha_mode.lower()
        self.alpha_init = float(alpha_init)
        if self.alpha_mode == "global":
            self.alpha_param = nn.Parameter(th.tensor(self.alpha_init)) if learn_alpha else None
            self.alpha_mlp = None
            self.alpha_pool = None
        else:
            self.alpha_param = None
            self.alpha_pool = alpha_pool.lower()
            assert self.alpha_pool in {"mean", "max"}
            self.alpha_mlp = nn.Sequential(
                nn.LayerNorm(self.d_embed),
                nn.Linear(self.d_embed, alpha_mlp_hidden),
                nn.ReLU(),
                nn.Linear(alpha_mlp_hidden, 1),
            )
        self._alpha_last = self.alpha_init

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

    def _embed(self, x: th.Tensor) -> th.Tensor:
        """
        Tworzy osadzenia (embeddings) dla każdej cechy z wejściowego tensora obserwacji.
        
        Wejście: x [B, N] - batch obserwacji, gdzie N to liczba cech.
        Wyjście: E [B, N, d_embed] - osadzenia dla każdej cechy.
        """
        if self.use_content_embed:
            # Tryb z osobnymi embedderami: każda cecha ma własną liniową warstwę
            # Dzielimy wejście na kolumny (po jednej cechy) i przekształcamy każdą osobno
            cols = [x.narrow(1, i, 1) for i in range(self.n_metrics)]
            # Stosujemy odpowiedni embedder do każdej kolumny i sklejamy wyniki
            E = th.stack([layer(col) for layer, col in zip(self.embedders, cols)], dim=1)
        else:
            # Tryb bez embedderów: skalowanie wartości przez stały wektor projekcji
            # Rozszerzamy x do [B, N, 1] i mnożymy przez [1, d_embed] -> [B, N, d_embed]
            E = x.unsqueeze(-1) * self.value_projection
        
        # Dodajemy osadzenia pozycyjne (positional encodings) jeśli są włączone
        # P_idx to parametry przypisane do każdej pozycji cechy (niezależne od wartości)
        if self.P_idx is not None:
            E = E + self.P_idx.view(1, self.n_metrics, self.d_embed)
        
        # Normalizacja warstwowa na końcu: stabilizuje uczenie i normalizuje osadzenia
        return self.ln_e(E)

    def _alpha(self, E: th.Tensor) -> th.Tensor:
        B = E.size(0)
        if self.alpha_mode == "global":
            if self.alpha_param is not None:
                val = th.sigmoid(self.alpha_param)
            else:
                val = th.tensor(self.alpha_init, device=E.device)
            return val.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        pooled = E.amax(dim=1) if self.alpha_pool == "max" else E.mean(dim=1)
        scalar = th.sigmoid(self.alpha_mlp(pooled))
        return scalar.view(B, 1, 1, 1)

    def _attention_logits(self, E: th.Tensor) -> th.Tensor:
        # B = rozmiar batcha, N = liczba cech, H = liczba głów
        B = E.size(0)
        # Zapytania/klucze zależne od wartości wejściowych (część dynamiczna)
        q_c = self.Wq_c(E).view(B, self.n_metrics, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        k_c = self.Wk_c(E).view(B, self.n_metrics, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        # Zapytania/klucze indeksowe (priorytet przypisany do cechy niezależnie od stanu)
        q_i = self.Q_idx.unsqueeze(0).expand(B, -1, -1, -1)
        k_i = self.K_idx.unsqueeze(0).expand(B, -1, -1, -1)
        # α decyduje, jak łączymy część dynamiczną i indeksową
        alpha = self._alpha(E)
        q = alpha * q_c + (1.0 - alpha) * q_i
        k = alpha * k_c + (1.0 - alpha) * k_i
        # Skaluje produkt QK standardowo przez sqrt(d_k)
        scores = th.matmul(q, k.transpose(-2, -1)) / (float(self.d_k) ** 0.5)
        # Temperatura uwagi pozwala regulować „ostrość” softmaxu
        return scores / max(1e-6, self.attn_temp)

    def _normalize_heads(self, scores: th.Tensor) -> th.Tensor:
        # Zawsze używamy row_softmax dla trybu generalized
        return F.softmax(scores, dim=-1)

    def _aggregate_heads(self, heads: th.Tensor) -> th.Tensor:
        """
        Agreguje wiele głów uwagi w jedną macierz uwagi.
        
        Wejście: heads [B, H, N, N] - macierze uwagi z każdej głowy (H głów).
        Wyjście: A [B, N, N] - pojedyncza macierz uwagi po agregacji.
        """
        # Obsługa przypadku z jedną głową: wybieramy tę głowę
        if self.n_heads == 1:
            A = heads[:, 0]
        # Średnia arytmetyczna wszystkich głów: uśrednia wagi uwagi z każdej głowy
        elif self.head_agg == "mean":
            A = heads.mean(dim=1)
        # Suma wszystkich głów: sumuje wagi z każdej głowy
        elif self.head_agg == "sum":
            A = heads.sum(dim=1)
        # Maksimum po głowach: wybiera największą wagę uwagi dla każdej pary cech
        else:
            A, _ = heads.max(dim=1)
        # Renormalizacja dla sum i max: zapewnia, że wiersze sumują się do 1
        # (dla "mean" nie jest potrzebna, bo średnia z znormalizowanych macierzy jest już znormalizowana)
        # row_softmax był już zastosowany wcześniej, ale sum/max może zmienić sumy wierszy
        if self.head_agg in {"sum", "max"}:
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return A

    def _diagnostics(self, A: th.Tensor, x: th.Tensor) -> None:
        with th.no_grad():
            # Używamy średniej po wierszach macierzy uwagi
            vec = A.mean(dim=1)
            metric = vec / vec.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            contrib = A.abs().sum(dim=1) * x.abs()
            contrib = contrib / contrib.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            self.metric_importance = metric.detach()
            self.contrib_importance = contrib.detach()
            self.attn_history.append(self.metric_importance.mean(dim=0).cpu())
            self.contrib_history.append(self.contrib_importance.mean(dim=0).cpu())
            self.total_steps += 1

    def _forward_batch(self, x: th.Tensor) -> th.Tensor:
        """
        Główna funkcja forward dla batcha obserwacji.
        
        Wejście: x [B, N] - batch obserwacji, gdzie N to liczba cech.
        Wyjście: out [B, N] lub [B, final_out_dim] - przekształcone cechy.
        """
        # Maska aktywnych cech: wyłącza niektóre cechy (ustawiane zewnętrznie)
        # Pomaga w eksperymentach z selekcją cech - maskowane cechy są zerowane
        if hasattr(self, "active_mask"):
            x = x * self.active_mask.to(x.device).view(1, -1)
        
        # Krok 1: Tworzenie osadzeń dla każdej cechy
        # x [B, N] -> E [B, N, d_embed]
        E = self._embed(x)
        
        # Krok 2: Obliczanie logitów uwagi i normalizacja
        # E [B, N, d_embed] -> heads [B, H, N, N] (H głów uwagi)
        heads = self._normalize_heads(self._attention_logits(E))
        
        # Krok 3: Agregacja wielu głów w jedną macierz uwagi
        # heads [B, H, N, N] -> A [B, N, N]
        A = self._aggregate_heads(heads)
        
        # Zapis macierzy uwagi do późniejszej analizy (bez gradientu)
        self.attn_matrix = A.detach()
        
        # Krok 4: Mieszanie cech przez macierz uwagi
        # A [B, N, N] * x [B, N, 1] -> mixed [B, N]
        # Każda cecha wyjściowa to ważona suma wszystkich cech wejściowych
        mixed = th.bmm(A, x.unsqueeze(-1)).squeeze(-1)
        
        # Krok 5: Połączenie rezydualne (domyślnie wyłączone, we wszystkich eksperymentach było wyłączone aby Attention bezpośrednio wpływała na wejscie do sieci LSTM)
        # Dodaje oryginalne wejście z wagą, co pomaga w uczeniu głębokich sieci
        if self.use_residual:
            mixed = x + self.residual_weight * mixed
        
        # Krok 6: Zbieranie statystyk diagnostycznych (ważność cech, wkłady)
        # Działa w trybie no_grad, nie wpływa na gradienty
        self._diagnostics(A, x)
        
        # Krok 7: Post-processing wyjścia
        if self.post_proj is not None:
            # Jeśli final_out_dim != N: projekcja do innego wymiaru + opcjonalna aktywacja
            out = self.post_proj(mixed)
        else:
            # Jeśli final_out_dim == N: opcjonalna normalizacja i aktywacja bezpośrednio na mixed
            out = mixed
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





def train_model(env, args):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    policy_kwargs = dict(
    features_extractor_class=AttentionExtractor,
    features_extractor_kwargs=dict(
        d_embed=7,
        d_k=4,
        n_heads=2,
        head_agg="mean",
        alpha_mode="global",
        alpha_init=0.5,
        learn_alpha=False,
        use_posenc=True,
        use_content_embed=False,
        attn_temp=1.2,
        final_out_dim=7,
        out_layernorm=False,
        out_activation="tanh",
        use_residual=False,
    ),
    lstm_hidden_size=64,
    n_lstm_layers=1,
    net_arch=[dict(pi=[16, 16], vf=[16,16])],
    activation_fn=th.nn.ReLU,
    ortho_init=True,
    normalize_images=False,
)
    #     net_arch=[dict(pi=[32, 32], vf=[32, 32])],
    # activation_fn=nn.ReLU,
    # ortho_init=True,
    
  




    if args.policy == 'MlpLstmPolicy':
        model = RecurrentPPO(
                "MlpLstmPolicy",
                #CriticOnlyAttentionRecurrentPolicy,  # <-- use the subclass
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
             #policy_kwargs=policy_kwargs,
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
                        #policy_kwargs=policy_kwargs,
                        tensorboard_log="./output_malota/gawrl_mlp")
    return model




#learning_rate=0.0003, # 0.00003
                        # vf_coef=0.5,
                        # clip_range_vf=1.0,
                        # max_grad_norm=0.5,
                        # n_epochs=10,
                        # gamma=0.99,
                        # ent_coef=0.01,
                        # clip_range=0.05,
                        # verbose=1,
                        #     seed=int(args.seed) if args.seed else 42,
                        #     tensorboard_log='./output_malota/'


