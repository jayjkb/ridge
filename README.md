# RIDGE

RIDGE (**Residual-Informed Diagnosis from Graph-Structured Evidence**) is a two-stage temporal graph neural network architecture for root cause analysis (RCA) in computer networks. A probabilistic emulator forecasts the next telemetry snapshot in a fault-free network. Comparing the observations with that forecast produces residuals in units of the emulator's predictive uncertainty. An RCA model reads histories of these residual graphs, ranks candidates comprising the no-fault case, devices, and links, and classifies the fault category. This repository implements the architecture as a six-stage pipeline, from data generation through evaluation.

## Reported Results

The dataset of 4,000 episodes, the trained checkpoints, and the evaluation outputs behind every reported number are archived at <https://doi.org/10.5281/zenodo.21966579>. Stages 2 to 6 below rebuild the intermediate datasets from those episodes and reproduce the values given under [Expected values](#expected-values).

## Environment

Python 3.12.3 with the CUDA 12.8 PyTorch build, managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync # creates .venv with Python 3.12.3 and the locked packages
source .venv/bin/activate # or prefix commands with `uv run`
```

`uv.lock` reproduces the recorded environment. Its pins are the `constraint-dependencies` set in [`pyproject.toml`](pyproject.toml), and [`.python-version`](.python-version) fixes the interpreter version.

```bash
python --version # 3.12.3
python -c 'import torch; print(torch.__version__, torch.version.cuda)' # 2.11.0+cu128 12.8
```

Every stage reads and writes under `artifacts/`, a symlink included in this repository. Repoint it at the directory that holds the dataset and the model outputs when running this repo yourself.

```bash
ln -sfn /path/to/artifact-root artifacts
```

## Stage 1: Generating the Dataset

Reproducing the reported results does not require this step, because the dataset is archived and downloadable. Generating a new one additionally requires Mininet, Open vSwitch, `tc`, FRRouting, D-ITG, `iperf3`, and network-administration privileges.

```bash
sudo ./src/scripts/install_ubuntu_deps.sh
./src/scripts/check_system_deps.sh
```

The archived dataset was generated with the command below. Arguments that are not shown keep their built-in values.

```bash
sudo taskset -c 0-13 .venv/bin/python -m ridge generate \
  --runs 4000 --workers 14 --output artifacts/stage1-dataset \
  --duration 180 --interval 2 --burst-mean-gap-sec 2.5 \
  --traffic-flow-min 20 --traffic-flow-max 24 --warmup-sec 60 \
  --fault-start-offset-min-sec 35 --fault-start-offset-max-sec 70 \
  --fault-duration-min-sec 25 --fault-duration-max-sec 45 \
  --min-training-runs 0
```

The run produced 2,000 healthy episodes and 500 for each of `drain`, `fiber_cut`, `link_degradation`, and `link_flap`, with no failed episodes. It took roughly 27 to 30 hours and occupies 9.4 GiB. The fully resolved configuration is recorded in [`configs/stage1_generation_profile.json`](configs/stage1_generation_profile.json).

## Stages 2 to 6: Reproducing the Results

The commands below run in order. Every training setting reported in the research project is already the built-in value of its command, so only the paths, the residual mode, the no-graph option, and the seed have to be given. The command `python -m ridge <command> --help` lists the remaining arguments.

```bash
# Stage 2: normal-emulator windows, train-only normalization, run-level splits
python -m ridge build-normal \
  --dataset-dir artifacts/stage1-dataset --output-dir artifacts/stage2-normal

# Stage 3: train the emulator, then evaluate it once on the test split
python -m ridge train-normal \
  --data-dir artifacts/stage2-normal --output-dir artifacts/stage3-emulator
python -m ridge evaluate-normal \
  --data-dir artifacts/stage2-normal \
  --checkpoint artifacts/stage3-emulator/best_normal_emulator.pt \
  --output artifacts/stage3-emulator/normal_evaluation.json # Table 1

# Stage 4: standardized residuals, and the matched raw-telemetry arm
python -m ridge build-residual \
  --dataset-dir artifacts/stage1-dataset --normal-data-dir artifacts/stage2-normal \
  --normal-emulator artifacts/stage3-emulator/best_normal_emulator.pt \
  --output-dir artifacts/stage4-residual-standardized
python -m ridge build-residual \
  --dataset-dir artifacts/stage1-dataset --normal-data-dir artifacts/stage2-normal \
  --residual-mode raw --output-dir artifacts/stage4-residual-raw

# Stage 5: three learned arms, each under seeds 42, 43 and 44
python -m ridge train-ridge --data-dir artifacts/stage4-residual-standardized \
  --output-dir artifacts/stage5-rca-standardized-seed42 --seed 42
python -m ridge train-ridge --data-dir artifacts/stage4-residual-raw \
  --output-dir artifacts/stage5-rca-raw-seed42 --seed 42
python -m ridge train-ridge --data-dir artifacts/stage4-residual-standardized \
  --output-dir artifacts/stage5-rca-nograph-seed42 --seed 42 --no-graph

# Stage 6: evaluate each checkpoint once, plus the threshold baseline
python -m ridge evaluate-ridge \
  --data-dir artifacts/stage4-residual-standardized \
  --checkpoint artifacts/stage5-rca-standardized-seed42/best.pt \
  --output artifacts/stage6-evaluations/standardized_seed42/test_evaluation.json
python -m ridge evaluate-threshold \
  --data-dir artifacts/stage4-residual-standardized \
  --output artifacts/stage6-threshold-baseline/threshold_evaluation.json

# Aggregate the three seeds of an arm into the mean and standard deviation
python src/scripts/aggregate_seed_metrics.py \
  artifacts/stage6-evaluations/standardized_seed4{2,3,4}/test_evaluation.json \
  --output artifacts/stage6-aggregates/standardized_aggregate.json # Table 2
```

Repeating the Stage 5 and Stage 6 commands with `--seed 43` and `--seed 44` for each of the three comparison methods gives nine checkpoints and nine evaluations. The trainers use the training and validation splits only. The test split is read by `evaluate-normal`, `evaluate-ridge`, and `evaluate-threshold` alone, each run once per checkpoint.

## Accepted Artifacts

| Stage | Path under `artifacts/` | Key parameters |
|---|---|---|
| 1 raw dataset | `stage1-dataset` | seed 42, 14 workers; 4,000 episodes (2,000 none, 500×4 faults) |
| 2 normal dataset | `stage2-normal` | `H_E=6`, horizon 1; 209,145 windows; split 2800/600/600 |
| 3 emulator | `stage3-emulator/best_normal_emulator.pt` | trained on `stage2-normal`, best-validation checkpoint |
| 4 residual (standardized) | `stage4-residual-standardized` | `H_R=6`; 316,000 windows; uses the Stage 3 checkpoint |
| 4 residual (raw) | `stage4-residual-raw` | same windows, labels and splits, emulator skipped |
| 5 learned arms | `stage5-rca-{standardized,raw,nograph}-seed{42,43,44}/best.pt` | 9 checkpoints |
| 6 baseline | `stage6-threshold-baseline/threshold_evaluation.json` | κ = 1.2227, calibrated on validation |
| 6 aggregates | `stage6-aggregates/{standardized,raw,nograph}_aggregate.json` | mean ± std over 3 seeds |

Stage 2 and every stage after it compare the recorded properties of their inputs. An incompatible feature schema, history length, prediction horizon, residual mode, category label set, or candidate catalog is reported by name and stops the run before any weights are loaded.

## Expected Values

Emulator forecasting on the test split, in normalized units:

| Family | RMSE | MAE | NLL | ±2σ coverage |
|---|---|---|---|---|
| Node | 0.337 | 0.044 | −4.30 | 0.995 |
| Link | 0.230 | 0.042 | −4.05 | 0.982 |
| Probe (continuous) | 0.496 | 0.037 | −3.01 | 0.997 |

The binary probe channels give a cross-entropy of 0.013.

Root-cause localization, detection, and category performance on the test split. The learned methods are reported as the mean ± sample standard deviation over three training seeds. The threshold baseline is deterministic, and it predicts no category.

| Method | F1 | Acc@1 | MRR | Node@1 | Link@1 | Cat-F1 |
|---|---|---|---|---|---|---|
| Threshold baseline | 0.579 | 0.874 | 0.915 | 0.736 | 0.219 | — |
| Raw-telemetry | 0.888±.010 | 0.976±.001 | 0.985±.001 | 0.715±.003 | 0.851±.020 | 0.845±.005 |
| No-graph | 0.925±.002 | 0.978±.001 | 0.987±.000 | 0.749±.002 | 0.856±.007 | 0.849±.022 |
| RIDGE (full) | 0.922±.003 | 0.983±.000 | 0.990±.000 | 0.742±.003 | 0.921±.003 | 0.866±.012 |

The per-category F1 scores of the full model are 0.980 for the no-fault class, 0.950 for `fiber_cut`, 0.858 for `link_degradation`, 0.831 for `drain`, and 0.709 for `link_flap`. The full model scores the test windows at roughly at 1,560 windows per second averaged over the three seeds, or 0.64 ms per window. That figure was measured on the 16-core CPU host used for training and does not carry over to other machines.

## Research Tools

Six Streamlit explorers inspect the artifact produced by each stage:

```bash
streamlit run src/apps/stage1_dataset_explorer.py -- --dataset-root <stage1>
streamlit run src/apps/stage2_normal_dataset_explorer.py -- --dataset-root <stage2>
streamlit run src/apps/stage3_normal_training_explorer.py -- --model-root <stage3>
streamlit run src/apps/stage4_residual_dataset_explorer.py -- --dataset-root <stage4>
streamlit run src/apps/stage5_ridge_training_explorer.py -- --model-root <stage5>
streamlit run src/apps/stage6_evaluation_explorer.py -- --model-root <stage6>
```

Three additional scripts analyze a Stage 1 dataset. They check integrity, cadence, and labels, quantify the telemetry effect of each fault, and write tables and figures on root-cause readiness.

```bash
python src/analysis/stage1_eda.py --dataset-root <stage1>
python src/analysis/stage1_fault_effects.py --dataset-root <stage1>
python src/analysis/stage1_rca_readiness.py --dataset-root <stage1>
```

## Quality Checks

```bash
ruff check .
ruff format --check .
python -m compileall -q src
```

## License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)** - see the [LICENSE](LICENSE) file for details.