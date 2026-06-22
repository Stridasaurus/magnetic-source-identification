# Magnetic Source Identification — Project 1 (Settgast)

This folder is the research record (and git repo) for **MTH 5320 — Project 1**:
identifying the 3D position and pole type (monopole vs. dipole) of a simulated
ionospheric magnetic source from a 29-station global magnetometer array, using a
multi-task feedforward neural network. It was split out of the original monolithic
notebook (`Copy of Strider_Settgast_Project_1.ipynb`, kept in the parent `Downloads` folder).

Repo support files: `README.md` (human landing page), `requirements.txt` (pinned env,
Python 3.12), `.gitignore` (excludes `*.pt`/`*.db` artifacts), `L058.txt` (29-station
sensor input), `Settgast_Project1_Report.md` (written report).

## Contents

| Notebook | Purpose | Dataset |
|----------|---------|---------|
| `00_Setup_and_Simulation.ipynb` | Optional overview / sanity check: install, simulator class, sensor loader, 3D array visualization. | demo grid only |
| `01_Experiment_1.ipynb` | Broad architecture search (Optuna, 14 hyperparameters). | 400 positions × 20 noise = 16,000 |
| `02_Experiment_2.ipynb` | Loss/optimizer focus; locks Exp 1 winners. Same dataset as Exp 1. | 400 positions × 20 noise = 16,000 |
| `03_Experiment_3.ipynb` | Larger dataset + grad_clip fix; full test-set evaluation and visualization. | 3,200 positions × 10 noise = 64,000 |

The companion writeup is `Settgast_Project1_Report.md` (in this repo).

## Each experiment notebook is self-contained

`01`, `02`, and `03` each re-include the install/upload cell, the simulator class, and
the sensor loader, so any of them runs on its own — you do **not** need to run
`00_Setup` first. (`00_Setup` is just an optional overview + array sanity check.)

Internal layout of each experiment notebook (markdown section hierarchy):

- **0 · Environment Setup** — `!pip install optuna tqdm` + `L058.txt` upload
- **1 · Simulation & Model Infrastructure** *(shared, identical across all three)* —
  1.1 simulator (MTSS), 1.2 sensor loader, 1.3 multi-task network, 1.4 Optuna helpers
- **2 · Dataset Generation** *(experiment-specific params)* — 2.1 generate, 2.2 preprocess
- **3 · Experiment N** — the Optuna study + final training
- (Exp 3 only) **4 · Test-Set Evaluation** (4.1 metrics, 4.2 visualizations) and
  **5 · Analysis & Conclusions**

## How to run (Google Colab)

Each notebook gets its own Colab runtime — state does **not** carry between notebook
files, which is why each experiment is self-contained.

1. Upload the notebook to [colab.research.google.com](https://colab.research.google.com)
   (File → Upload notebook).
2. Runtime → Change runtime type → **T4 GPU**.
3. Run the first cell — it installs deps and loads **`L058.txt`** (uses a copy in the
   working directory if present, otherwise prompts for upload).
4. Runtime → **Run all**. Optuna writes `optuna_experiment*.db`; the trained model is
   saved as `magnetic_model_experiment*.pt` (download from the file panel before the
   session expires). Exp 3's evaluation cell loads `magnetic_model_experiment3.pt`,
   which its own training cell produces earlier in the same run.

Reproducibility: data generation (`seed 42` train grid, `seed 123` test grid), the
train/val split (`random_state=42`), `torch.manual_seed(42)`, and the Optuna sampler
(`TPESampler(seed=42)`) are all seeded, so the search is reproducible from a clean run;
metrics may still vary slightly due to GPU floating-point nondeterminism.

## Key facts / conventions (so edits stay consistent)

- **Source shell:** sources sit on a spherical shell at **800–3,100 km** altitude above
  Earth's surface (geocentric radii 7,171–9,471 km). Code: `min_alt = EARTH_RADIUS + 800`,
  `max_alt = EARTH_RADIUS + 3100`.
  - Note: the report PDF currently states 400–3,100 km; the notebooks/code use 800 km,
    and the notebook markdown was deliberately set to 800 to match the code.
- **Inputs:** 29 sensors × 3 field axes (Bx, By, Bz) → 87 features per sample.
- **Targets:** `[X, Y, Z, is_monopole, is_dipole]`.
- **Preprocessing:** `RobustScaler` per axis on inputs; `MinMaxScaler` to [0,1] on
  positions; one-hot → integer class labels for `CrossEntropyLoss`.
- **Dataset trade-off:** noise realizations are correlated augmentation; unique positions
  add independent spatial coverage. Exp 3 trades the former for the latter.
- **Final model (Exp 3):** test type accuracy ≈ 97.64%, position RMSE ≈ 0.1874 (scaled).
- **Known code note:** the base `ModelTrainer.train_epoch()` hardcodes `max_norm=1.0`,
  so `grad_clip` tuning had no effect in Exp 2. Exp 3 fixes this via the `TunableTrainer`
  subclass that applies `self.grad_clip`.

## Editing rules

- The split notebooks intentionally diverge from the master in a few sanctioned ways:
  per-experiment `n_grid_points`/`noise_realizations`, a robust `L058.txt` loader (repo
  copy or upload), `torch`/`numpy` + `TPESampler` seeding, and a `device` definition added
  to Experiment 2 (the master relied on Experiment 1's cell setting it). Otherwise, do
  **not** alter executable logic when refactoring — only split and copy.
- Preserve existing cell outputs exactly; the one sanctioned exception is the
  dataset-generation cell when its params are changed (Exp 1, Exp 2, and the Setup demo
  grid have that one cell's stale output cleared, since it will be re-run).
- Validate notebooks are valid JSON after any programmatic edit.
- When editing notebook JSON via Python, run from a **script file** (not `python -c "..."`
  in bash) — backticks in markdown get eaten by bash command substitution otherwise.
