## Automatic choice of metrics
Głównym skryptem projektu jest `run.py` uruchamiany za pomocą komendy:
```
python run.py --env ThreeSizeAppEnv-v1 --policy MlpLstmPolicy --algo RecurrentPPO --num_timesteps 700_000  --tensorboard_log '.' --model_name 'mlplstm_64_0_1_2_5_6' --queue_wait_penalty 0.0005 --mips_per_core 4400 --workload_file "TEST-DNNEVO-2.swf" --num_env 1 --simulator_speedup 60.0   
```


### Feature selector

`feature_selector_2.py` - klasa selektora cech, wykorzystuje ekstraktor zaimplementowany w `attention.py` oraz obiekty `Callback` z pliku `callbacks.py`
`attention.py` - plik z modułem Attention
`callbacks.py` - plik z definicją Callbacków do uczenia PPO

### Analiza wybieranych cech

`attention_analysis/attention_spca_ig_metric_selection.ipynb` - notatnik z procesem selekcji cech na podstawie danych zebranych w rezerwuarze oraz danych zebranych w ewaluacji

`training_analysis_tensorboard.ipynb` - notatnik z analizą przebiegów uczenia z tensorboardu

`attention_analysis/parameter_count.ipynb` - notatnik z analizą liczby parametrów modeli

### Ogólna struktura projektu

FINAL_MODELS - folder zawierający artefakty uczonych modeli
logs/ - folder z logami z Callbacków (wykorzystywanymi do obliczania rankingów cech na podst. SPCA/IG/Attention)