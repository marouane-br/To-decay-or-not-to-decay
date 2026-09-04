# To-decay-or-not-to-decay

## Project Overview

This repository contains the completed experiment pipeline for a paper on decaying Gaussian noise schedules for denoising-autoencoder-based intrusion detection systems (IDS). The study compares constant-noise reference models against decaying Gaussian noise schedules using paired configurations for statistical analysis.

The implemented schedules are:
- constant-noise reference (`CONSTANT` in code, commands, and exported labels), and
- decaying Gaussian noise schedules (`LINEAR`, `EXPONENTIAL`, `COSINE`, `FIBONACCI`, `SIGMOID`, `CAUCHY`, `LAPLACE`, `LOGISTIC`).

All generated artifacts are written under `generated_outputs/`.

## Repository Contents

- `main.py`: experiment runner for dataset loading, fixed splits, model training, validation-threshold selection, and metric export.
- `statistical_analysis.py`: paired and family-level statistical analysis for exported metrics.
- `requirements.txt`: Python dependencies.
- `LICENSE`: Apache License 2.0 for the repository code.

## Supported Datasets

Supported datasets:
- `NSL_KDD`
- `UNSW_NB15`
- `CTU13`
- `HIKARI2021`

## Dataset Directory and `DATASET_DIR`

Default dataset directory:

```text
dataset/
```

You can override it without changing code:

```bash
export DATASET_DIR=/path/to/dataset
```

The original datasets are not redistributed in this repository. Place each dataset in the expected local dataset directory before running experiments.

## Supported AE Variants

- `FF_DAE`: feedforward denoising autoencoder.
- `DVAE`: denoising variational autoencoder using ELBO-style anomaly scoring; controlled by `--vae_beta`.
- `RES_DAE`: residual dense denoising autoencoder.
- `SPARSE_DAE`: sparse denoising autoencoder with L1 activity regularization; controlled by `--sparsity_l1`.

## Main CLI Usage

```bash
python main.py [options]
```

Key options:
- `--dataset` (single dataset, multiple datasets, or omitted for all configured datasets)
- `--schedules` (`ALL` or a subset of schedule families)
- `--seeds` (comma-separated list; default when omitted is `42,7,123`)
- `--ae_variant` (`all`, a single variant, or comma-separated variants)
- `--noise_type` (default `gaussian`)
- `--vae_beta` (DVAE only)
- `--sparsity_l1` (SPARSE_DAE only)
- `--resume`
- `--no_plots`
- `--save_epoch_recon` / `--no-save_epoch_recon`

## Noise Schedule Families

Supported schedule families:
- `CONSTANT` (constant-noise reference)
- `LINEAR`
- `EXPONENTIAL`
- `COSINE`
- `FIBONACCI`
- `SIGMOID`
- `CAUCHY`
- `LAPLACE`
- `LOGISTIC`

Schedule grid behavior:
- `CONSTANT` uses a sigma grid.
- Decaying families use the shared `R1` to `R36` start/end noise range grid.

## Smoke Test Commands

These commands validate that the runner starts with a minimal subset. They do not rerun the completed full experiment grid.

Single variant smoke tests:

```bash
python main.py --dataset NSL_KDD --seeds 7 --no_plots --ae_variant ff_dae --schedules CONSTANT LINEAR --no-save_epoch_recon
python main.py --dataset NSL_KDD --seeds 7 --no_plots --ae_variant dvae --vae_beta 0.001 --schedules CONSTANT LINEAR --no-save_epoch_recon
python main.py --dataset NSL_KDD --seeds 7 --no_plots --ae_variant res_dae --schedules CONSTANT LINEAR --no-save_epoch_recon
python main.py --dataset NSL_KDD --seeds 7 --no_plots --ae_variant sparse_dae --sparsity_l1 1e-5 --schedules CONSTANT LINEAR --no-save_epoch_recon
```

Multiple variants smoke test:

```bash
python main.py --dataset NSL_KDD --seeds 7 --no_plots --ae_variant res_dae,sparse_dae --sparsity_l1 1e-5 --schedules CONSTANT LINEAR --no-save_epoch_recon
```

Comma-separated variant examples:
- Correct: `--ae_variant res_dae,sparse_dae`
- Correct: `--ae_variant "res_dae,sparse_dae"`
- Incorrect: `--ae_variant res_dae, sparse_dae`

## Full Run Examples

Single dataset, all active variants:

```bash
python main.py --dataset NSL_KDD --seeds 42,7,123 --ae_variant all --noise_type gaussian --resume
```

All supported datasets, all active variants:

```bash
python main.py --dataset NSL_KDD UNSW_NB15 CTU13 HIKARI2021 --seeds 42,7,123 --ae_variant all --noise_type gaussian --resume
```

These runs can be computationally expensive because each selected schedule family expands into multiple fixed configurations. Do not rerun the full grid unless you intend to regenerate experiment artifacts.

## Output Structure

```text
generated_outputs/
  runs/
    <DATASET>/
      <DATASET>_seed_<SEED>/
        splits/
        <AEVariant>_seed_<SEED>/
          metrics/
          plots/
          run_info/
  statistical_outputs/
```

Important metrics file patterns under `metrics/`:
- `*_final_test_metrics_seed<SEED>.csv`
- `*_train_val_metrics_seed<SEED>.csv`
- `*_selected_threshold_rows_seed<SEED>.csv`

Exact dataset prefixes are included in filenames by the runner. Keep these filenames and schemas unchanged when reproducing the statistical analysis.

## Statistical Analysis Command

Example command:

```bash
python statistical_analysis.py --input_dir generated_outputs --analysis_mode both --metric MCC --datasets NSL_KDD --ae_variants FF_DAE,DVAE,RES_DAE,SPARSE_DAE --no_plots
```

Notes:
- Paired mode is the primary inferential layer.
- Family mode is secondary/complementary.
- `MCC` is the default primary metric.
- Paired summaries report Holm and BH corrected p-values.
- Family-level analysis uses ANOVA and Tukey HSD.

## Reproducibility Notes

Deterministic model initialization seeds are derived from dataset, seed, AE variant, and base seed. The project is designed for reproducible experiment setup and comparable reruns, but it does not claim bitwise-identical results across all hardware/software environments.

Resume mode can be enabled with `--resume`. Completed `ConfigID` runs are skipped; incomplete or corrupted `ConfigID` runs are cleaned and rerun from scratch using the same deterministic initialization seed. Training is not resumed from partial Keras checkpoints.


Example:

```bash
export DATASET_DIR=/path/to/dataset
python main.py --dataset NSL_KDD --seeds 7 --no_plots --ae_variant ff_dae --schedules CONSTANT LINEAR --no-save_epoch_recon --resume
python statistical_analysis.py --input_dir generated_outputs --analysis_mode both --metric MCC --datasets NSL_KDD --ae_variants FF_DAE --no_plots
```

## License

This code is released under the Apache License 2.0. The original datasets are not redistributed and remain subject to their respective licenses and terms of use.

