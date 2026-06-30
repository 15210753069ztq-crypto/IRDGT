# IRDGT

Official research implementation of IRDGT for traffic accident risk
forecasting on the NYC and Chicago benchmarks.

The repository contains the training and evaluation pipeline, the IRDGT model,
the optional weather-conditioned subgraph generator, experiment
configurations, and the preprocessed datasets used by the code.

## Repository layout

```text
.
├── config/
│   ├── chicago/          # Chicago main and ablation configurations
│   ├── examples/         # Weather-subgraph examples
│   └── nyc/              # NYC main and ablation configurations
├── data/
│   ├── chicago/
│   └── nyc/
├── lib/                  # Data loading, losses, metrics, and early stopping
├── model/
│   └── HGCN.py           # IRDGT and optional weather-subgraph modules
├── train.py
└── requirements.txt
```

The weather-conditioned graph is implemented by
`WeatherConditionalGraphModule` in `model/HGCN.py`. It is intentionally kept
as an optional component and can be enabled with:

```json
{
  "use_weather": true,
  "use_weather_subgraph": true,
  "use_weather_as_relation": true,
  "use_weather_rank_head": true
}
```

Ready-to-run examples are provided under `config/examples/`.

## Data

The repository uses Git LFS for the two large `all_data.pkl` files. Install Git
LFS before cloning:

```bash
git lfs install
git clone https://github.com/15210753069ztq-crypto/IRDGT.git
cd IRDGT
git lfs pull
```

Dataset summary:

| Dataset | Period | Tensor shape | Interval |
|---|---|---:|---:|
| NYC | Jan--Dec 2013 | `(8760, 48, 20, 20)` | 1 hour |
| Chicago | Feb--Sep 2016 | `(5832, 41, 20, 20)` | 1 hour |

See `data/README.md` for the channel definitions and file descriptions.

## Environment

The release was checked with Python 3.10, PyTorch 2.2, NumPy 1.26, pandas
2.2, NetworkX 3.2, and Matplotlib 3.9.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Training

NYC:

```bash
python train.py --config config/nyc/GSNet_NYC_Config.json --gpus 0
```

Chicago:

```bash
python train.py --config config/chicago/DSHGNN_Chicago_CurrentBest_Config.json --gpus 0
```

Optional weather-conditioned subgraph:

```bash
python train.py --config config/examples/IRDGT_NYC_WeatherSubgraph_Config.json --gpus 0
```

Use `--test_mode --training_epoch 1` for a short pipeline check.

## Outputs

Checkpoints and generated predictions are written under `outputs/` and are
ignored by Git. When `save_predictions` is enabled, the best validation model
stores predictions and labels for later error analysis.

## Acknowledgement

The dataset organization and baseline components follow the GSNet benchmark:

> B. Wang, Y. Lin, S. Guo, and H. Wan, "GSNet: Learning Spatial-Temporal
> Correlations from Geographical and Semantic Aspects for Traffic Accident
> Risk Forecasting," AAAI, 2021.

Please cite the original benchmark and the IRDGT paper when using this
repository. The IRDGT citation will be added after publication.

## License

No open-source license has been assigned yet. Unless a license is added, the
code and data remain subject to their respective owners' rights.
