## Automatic choice of metrics

To repozytorium zawiera kod do przeprowadzenia treningu dla agentów Proximal Policy Optimization (PPO), sterujących zasobami chmurowymi w środowisku symulowanym (opartym o workloady w stylu `pytorch-dnnevo` / CloudSimPlus).  
Na bazie standardowej polityki RecurrentPPO + MLP/LSTM dodano **moduł automatycznego wyboru metryk (cech)**, który potrafi wskazać zredukowany zestaw metryk wejściowych bez pogorszenia jakości polityki.

Kod został opracowany w ramach pracy magisterskiej dotyczącej automatycznego wyboru metryk dla agentów DRL do zarządzania zasobami chmurowymi.


## Uruchomienie przykładowego treningu PPO

Głównym skryptem projektu jest `run.py` uruchamiany za pomocą komendy:
```
python run.py --env ThreeSizeAppEnv-v1 --policy MlpLstmPolicy --algo RecurrentPPO --num_timesteps 700_000  --tensorboard_log '.' --model_name 'mlplstm_64_0_1_2_5_6' --queue_wait_penalty 0.0005 --mips_per_core 4400 --workload_file "TEST-DNNEVO-2.swf" --num_env 1 --simulator_speedup 60.0   
```

---

## Moduł automatycznego wyboru metryk

Logika automatycznego wyboru metryk jest zaimplementowana w kilku plikach w katalogu `automatic_feature_selection`:

### `feature_selector.py`

Główny plik automatycznego wyboru metryk.  
Korzysta z rankingów generowanych przez inne moduły (IG, SPCA, Attention) i stosuje następującą procedurę:

- normalizuje wektory ważności cech,
- sortuje metryki według ważności,
- wybiera najmniejszy prefiks metryk, którego skumulowana ważność przekracza zadany próg (np. τ = 0.95),  
  z opcjonalnym minimalnym rozmiarem zbioru `K_min`.

---

### `ig_attribution.py`

Implementuje **Integrated Gradients** dla polityki PPO:

- Odtwarza zapisane trajektorie z końcowej fazy treningu.
- Dla każdej pary stan–akcja oblicza atrybucje IG względem wyjścia sieci aktora (np. logarytmu prawdopodobieństwa wybranej akcji) względem metryk wejściowych.
- Agreguje wartości bezwzględne IG w czasie, żeby uzyskać średnią ważność każdej metryki.

Ten plik odpowiada za wszystkie obliczenia związane z IG; `feature_selector.py` jedynie wczytuje jego wyniki.

---

### `spca.py`

Implementuje ranking metryk na podstawie **SparsePCA**:

- Traktuje bufor rezerwuarowy (zalogowane obserwacje) jako macierz danych `X ∈ R^{T×N}`.
- Wykonuje podstawowe przetwarzanie wstępne (np. skalowanie / „ucięcie” wartości odstających).
- Dopasowuje model `SparsePCA` z `K` składowymi i wyciąga:
  - ładunki składowych dla każdej metryki,
  - wariancję / „aktywność” każdej składowej.
- Łączy te informacje w jedną wartość ważności dla każdej metryki (np. ważoną sumę modułów ładunków), a następnie przekazuje je do `feature_selector.py`.

Daje to ranking metryk całkowicie oparty na danych, niezależny od konkretnego modelu polityki.

---

### `attention.py`

Definiuje **ekstraktor cech oparty na Attention**, używany jako `features_extractor_class` w Stable-Baselines3 / `sb3_contrib`:

- Implementuje klasę `AttentionExtractor(BaseFeaturesExtractor)`, która:
  - osadza każdą skalarną metrykę w wektorze wymiaru `d_embed`,
  - dodaje uczone osadzenia pozycyjne (na metrykę),
  - wyznacza hybrydowe zapytania i klucze
    (część zależna od wartości i część indeksowa),
  - wykonuje dot-product self-attention nad metrykami,
  - opcjonalnie stosuje połączenie rezydualne oraz projekcję wyjścia.
- Podczas treningu wyznacza i zapisuje:
  - `metric_importance` – średnią „uwagę” przypisaną każdej metrce,
  - `contrib_importance` – miarę wkładu metryki, uwzględniającą zarówno wagi uwagi, jak i amplitudę sygnału.
































### Analiza wybieranych cech

`attention_analysis/attention_spca_ig_metric_selection.ipynb` - notatnik z procesem selekcji cech na podstawie danych zebranych w rezerwuarze oraz danych zebranych w ewaluacji

`training_analysis_tensorboard.ipynb` - notatnik z analizą przebiegów uczenia z tensorboardu

`attention_analysis/parameter_count.ipynb` - notatnik z analizą liczby parametrów modeli
