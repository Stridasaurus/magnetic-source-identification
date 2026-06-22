# Magnetic Source Identification Using a Multi-Task Feedforward Neural Network

**Strider Settgast**
MTH 5320 — Project 1
June 22, 2026

---

## Abstract

This project investigates the capability of a fully-connected multi-task neural network to solve the magnetic inverse problem: given simultaneous field measurements from a 29-station global magnetometer array, predict the 3D position and pole type (monopole or dipole) of an ionospheric magnetic source. Data was generated using a custom Python-based Magnetometer Time Series Simulator (MTSS) that computes exact dipole and monopole field equations with additive Gaussian noise. Four sequential experiments were conducted: three Optuna-driven hyperparameter studies (architecture search, loss/optimizer tuning, and a dataset-coverage expansion) followed by a controlled architectural intervention that constrains the regression head to output positions on the source shell. Across every experiment the source-type classifier reached 97–98% accuracy, while the aggregate position RMSE plateaued near 0.187 (per-axis, normalized) — about 6,080 km of 3D error. The decisive finding is that this aggregate number conceals two very different sub-problems: **monopole** sources are localized well (~720 km, ~8% of the shell radius) and *do* improve with more data, whereas **dipole** sources sit at ~8,300 km — the geocentric-radius scale — in every experiment and never respond to data, architecture, or loss changes. The spherical-output experiment confirms the cause: it forbids the regression-toward-center hedge that is RMSE-optimal for ambiguous dipoles, leaving monopoles essentially unchanged but pushing dipole predictions to ~11,000 km, the mean separation of two random points on the shell. The residual error is therefore dominated not by model capacity but by the irreducible moment ambiguity of the dipole inverse problem under this sensor geometry.

---

## 1. Introduction and Goals

The magnetic inverse problem — inferring the location and character of a source from remote field measurements — is fundamental to geophysical and space physics research. In the ionosphere, current systems and plasma structures generate magnetic signatures detectable by ground-based magnetometer networks. Traditional inversion methods rely on solving the forward model analytically, which requires strong assumptions about source geometry and is sensitive to noise. Data-driven approaches offer a complementary path, potentially learning the inverse mapping directly from data without requiring an explicit analytical solution.

The goals of this study were:

1. Build a simulation pipeline capable of generating large, labeled datasets of realistic magnetic field observations from monopole and dipole sources distributed across an ionospheric shell.
2. Train a multi-task feedforward neural network to simultaneously predict source position (regression) and source type (classification) from raw sensor readings.
3. Use systematic hyperparameter optimization to identify the best architecture and training configuration for this problem.
4. Characterize the performance ceiling of the learned model and assess how close it approaches the theoretical Bayes error for this sensor geometry and noise level.

This study was conducted solo. All data generation, model design, training, and analysis were performed by the author.

---

## 2. Data

### 2.1 Simulation Setup

All data was generated using the MTSS, a custom Python simulator built on the magnetic dipole and monopole field equations. The simulator computes the three-component magnetic field vector (Bx, By, Bz) at each of 29 real-world magnetometer station positions loaded from a network configuration file (`L058.txt`). Sensor coordinates are converted from geographic to 3D Cartesian coordinates on a sphere of radius 6,371 km.

Magnetic sources are placed uniformly at random on a spherical shell between 800 and 3,100 km altitude above Earth's surface (geocentric radii of 7,171–9,471 km, mean radius ≈ 8,321 km), simulating ionospheric source conditions. Two source types are simulated:

- **Monopole:** an idealized point magnetic charge with a 1/r² isotropic field. Not physically realized in nature but conceptually useful as a limiting case: its field pattern is fully determined by position, so the inverse problem is well-posed.
- **Dipole:** a magnetic dipole with a randomly oriented moment vector and 1/r³ falloff. The dominant term in the multipole expansion of realistic ionospheric current systems. Because its field depends on both position *and* moment orientation, its inverse problem is ill-posed when the moment is unknown.

For each source position and type, the simulator draws a random magnetic moment magnitude uniformly from [10¹², 10¹⁴] A·m² and a random moment direction uniformly on the unit sphere. The clean field is computed exactly via the dipole or monopole equations. Additive Gaussian noise with standard deviation σ = 0.01 is then applied independently to each field component to simulate sensor noise.

### 2.2 Dataset Design

The dataset design evolved across experiments based on findings about what limits model performance. The final dataset used for Experiments 3 and 4 is described here; differences from earlier experiments are noted in Section 4.

**Training/validation set (noisy):**

| Parameter | Value |
|-----------|-------|
| Unique source positions | 3,200 |
| Source types | 2 (Monopole, Dipole) |
| Noise realizations per position | 10 |
| Total samples | 64,000 |
| Input shape | (64,000 × 87): 29 sensors × 3 axes |
| Target shape | (64,000 × 5): [X, Y, Z, is\_monopole, is\_dipole] |

**Test set (clean, noiseless):**

| Parameter | Value |
|-----------|-------|
| Unique source positions | ~5,647 |
| Noise realizations | 0 (noiseless) |
| Total samples | ~11,294 |

The decision to use a noiseless test set is intentional: it isolates localization error from noise sensitivity, giving the cleanest possible measure of how well the model has learned the inverse mapping.

A key design decision was the trade-off between unique positions and noise realizations. Noise realizations share the same underlying clean field and are therefore correlated — they build noise robustness but do not add independent spatial information. In Experiments 1 and 2, the dataset used 400 unique positions × 20 noise realizations = 16,000 samples. In Experiment 3, this was restructured to 3,200 unique positions × 10 noise realizations = 64,000 samples. This gives 8× more independent spatial coverage for 4× more total samples — a substantially better trade.

### 2.3 Preprocessing

Input features were scaled using a `RobustScaler` applied per magnetic axis (Bx, By, Bz treated jointly across all 29 sensors). `RobustScaler` was chosen over `StandardScaler` because the magnetic field magnitudes at ionospheric altitudes span several orders of magnitude and contain sensor-specific outliers. Scaling was fit on the training set only and applied identically to validation and test sets.

Position targets were scaled to [0,1] using a `MinMaxScaler` fit on training positions. Classification targets were converted from one-hot encoding to integer class labels (0 = monopole, 1 = dipole) for compatibility with PyTorch's `CrossEntropyLoss`.

The 82/18 train/validation split was performed with a fixed random seed (42) for reproducibility. The test set was generated with an independent random seed (123) to guarantee no position overlap with the training distribution.

---

## 3. Model Architecture

### 3.1 Multi-Task Network Design

A fully-connected multi-task neural network (`MultiTaskMagneticNet`) was designed to share a common backbone across both tasks, with separate task-specific heads for regression (position) and classification (type). This architecture reflects the intuition that position and type prediction draw on the same underlying representation of the field structure, and that sharing early layers should improve data efficiency.

The network has three components:

**Shared backbone:** a stack of fully-connected layers with batch normalization, dropout, and GELU activations. The backbone learns a shared embedding of the 87-dimensional input. Layer widths are controlled by a `hidden_dim` parameter and a `decay_factor` that tapers width across layers, allowing the search to explore both constant-width and funnel architectures.

**Regression head:** a smaller stack of fully-connected layers terminating in a 3-dimensional output representing the scaled [X, Y, Z] position. Uses half the dropout rate of the backbone. In Experiment 4 the final linear layer of this head is replaced by a spherical parametrization (Section 4.4).

**Classification head:** a single fully-connected layer terminating in a 2-dimensional logit output for monopole/dipole classification.

### 3.2 Loss Function

A combined multi-task loss was used:

```
L_total = w_loc × L_loc + w_type × L_type
```

where `L_loc` is a Huber (SmoothL1) loss on the scaled position prediction, `L_type` is cross-entropy on the type logits, and `w_loc`, `w_type` are tunable loss weights. The Huber loss `beta` parameter, which controls the transition between quadratic and linear loss behavior, was introduced as a tunable hyperparameter in Experiment 2.

### 3.3 Training Configuration

The model was trained with the AdamW optimizer with weight decay acting as L2 regularization. A `ReduceLROnPlateau` scheduler was used to reduce the learning rate by a factor of 0.7 when validation loss stopped improving for 10 consecutive epochs. Gradient clipping was applied to prevent instability. Early stopping with a configurable patience parameter prevented overfitting.

An important implementation note: an early bug in the code set a `grad_clip` attribute on the trainer object but the `train_epoch()` method still read the hardcoded `max_norm=1.0`. This was corrected in Experiment 3 via a `TunableTrainer` subclass that overrides `train_epoch()` to use `self.grad_clip`, making gradient clipping tuning effective for the first time.

---

## 4. Experiments

Experiments 1–3 used Optuna with the TPE (Tree-structured Parzen Estimator) sampler and a MedianPruner. Optuna's TPE algorithm learns from previous trials, concentrating search effort toward regions of the hyperparameter space that have produced good results; the MedianPruner stops underperforming trials early — those whose validation loss at a given epoch is worse than the median of all completed trials at that epoch — saving compute for more promising configurations. Each study ran 30 trials, stored results in a SQLite database for session persistence, and trained a final model with the best configuration for up to 400 epochs with early stopping. Experiment 4 is a single controlled run (no search) that changes only the output parametrization.

*Reproducibility note: data generation, the train/validation split, `torch.manual_seed(42)`, and the Optuna sampler (`TPESampler(seed=42)`) are all seeded, so each study is reproducible from a clean run. Reported metrics may still vary by a small margin due to GPU floating-point nondeterminism.*

### 4.1 Experiment 1 — Broad Architecture Search

**Dataset:** 400 unique positions × 2 types × 20 noise realizations = **16,000 samples**

The first experiment cast a wide net across all 14 hyperparameters with minimal prior assumptions. The goal was to understand which parameters matter and which don't before narrowing the search.

**Search space:**

| Parameter | Range |
|-----------|-------|
| `n_layers` | 2 – 5 |
| `hidden_dim` | 64 – 512 (step 32) |
| `decay_factor` | 0.70 – 1.00 |
| `n_loc_layers` | 1 – 3 |
| `n_type_layers` | 1 – 3 |
| `dropout_rate` | 0.10 – 0.50 |
| `weight_decay` | 1e-6 – 1e-3 (log) |
| `use_residual` | True / False |
| `batch_size` | 32 / 64 / 128 / 256 |
| `learning_rate` | 1e-4 – 5e-3 (log) |
| `bn_momentum` | 0.05 – 0.30 |
| `activation` | relu / silu / gelu |
| `loc_weight` | 0.50 – 2.00 |
| `type_weight` | 0.50 – 2.00 |

**Epochs per trial:** 50 | **Final model epochs:** 200 | **Early stopping patience:** 30

**Results:**

| Metric | Value |
|--------|-------|
| Best val loss | 0.1009 |
| Best test RMSE (per-axis, scaled) | 0.1915 |
| Best test type accuracy | 97.85% |

**Key findings:** Four parameters converged decisively across all 30 trials. `use_residual=False` appeared in every competitive trial — residual connections added complexity without benefit at this dataset size. `activation=gelu` was favored by every top trial. `batch_size=128` appeared in the winning cluster. `n_type_layers=1` dominated — a shallow classification head was consistently better.

Critically, the position RMSE sat between 0.190 and 0.193 across all 30 trials regardless of architecture. A 5-layer network with 512-wide hidden layers achieved nearly the same RMSE as a 2-layer network with 64-wide layers. This ~1.5% spread suggested the model's capacity was not the limiting factor — the data was.

### 4.2 Experiment 2 — Loss Function and Optimizer Focus

**Dataset:** Same 16,000 samples

Experiment 2 locked the four decisive parameters from Experiment 1 and introduced three new hyperparameters targeting the loss function and optimizer that were previously hardcoded.

**Locked parameters:** `use_residual=False`, `activation=gelu`, `batch_size=128`, `n_type_layers=1`

**New parameters introduced:**

| Parameter | Range | Purpose |
|-----------|-------|---------|
| `huber_beta` | 0.01 – 1.00 (log) | SmoothL1 loss transition point |
| `grad_clip` | 0.50 – 5.00 (log) | Gradient clipping threshold |
| `scheduler_factor` | 0.3 / 0.5 / 0.7 | LR decay aggressiveness |

All other ranges were tightened based on the Experiment 1 winning cluster.

**Epochs per trial:** 75 | **Final model epochs:** 400 | **Early stopping patience:** 25

**Results:**

| Metric | Value |
|--------|-------|
| Best val loss | 0.0642 |
| Improvement vs. Exp 1 | −36% |
| Test RMSE (per-axis, scaled) | ~0.191 (unchanged) |

**Key findings:** The best trial used `huber_beta=0.931` — near the upper bound of 1.0, meaning the loss behaved almost identically to MSE. With well-controlled Gaussian noise and normalized targets, the model benefits from the quadratic penalty that pushes hard on small residual errors in the late training phase. A gentle scheduler (`factor=0.7`) outperformed an aggressive one.

The most important finding was the decoupling of val loss and RMSE. Val loss dropped 36% but RMSE was completely stationary at ~0.191. The loss-function tuning improved the numerical optimization landscape without moving the actual prediction quality. The RMSE plateau is a property of the data, not the model or optimizer.

Note: the `grad_clip` tuning in this experiment had no actual effect due to a bug — the base trainer's `train_epoch()` read a hardcoded `max_norm=1.0` rather than `self.grad_clip`. This was discovered and corrected before Experiment 3.

### 4.3 Experiment 3 — Larger Dataset and Bug Fix

**Dataset:** 3,200 unique positions × 2 types × 10 noise realizations = **64,000 samples** (the Optuna study used a 1,600-position intermediate set; the final model was trained on the full 3,200 positions)

Experiment 3 addressed the data coverage ceiling identified in Experiments 1 and 2. The key change was replacing correlated noise realizations with independent spatial positions: halving realizations from 20 to 10 while multiplying unique positions, giving 8× more independent spatial coverage.

**Locked parameters:** all Experiment 2 winners plus `scheduler_factor=0.7`. **New:** `TunableTrainer` subclass overrides `train_epoch()` so `grad_clip` is actually applied. **Widened:** `dropout_rate` upper bound raised to 0.35 and `hidden_dim` to 512 (more data supports stronger regularization and wider networks).

**Epochs per trial:** 75 | **Final model epochs:** 400 | **Early stopping patience:** 50

**A note on the two runs (reproducibility benchmark).** Because Optuna's TPE search is stochastic, Experiment 3 was carried out twice as independent searches. **Run A** settled on a four-layer backbone, **Run B** on a three-layer backbone, yet the two produced essentially identical held-out performance. This informal reproducibility benchmark reinforces the central finding (Section 4.1) that the problem is data- and physics-limited rather than architecture-limited: the search can pick quite different network shapes of equal quality. Run B is the run embedded in the repository notebook.

**Best trial parameters — two independent Optuna runs:**

| Parameter | Run A (Trial 15) | Run B (Trial 23) |
|-----------|------------------|------------------|
| `hidden_dims` | [448, 432, 417, 402] | [448, 399, 356] |
| `n_loc_layers` | 3 | 2 |
| `dropout_rate` | 0.179 | 0.226 |
| `weight_decay` | 8.0e-4 | 1.8e-4 |
| `bn_momentum` | 0.166 | 0.202 |
| `learning_rate` | 1.86e-4 | 1.48e-4 |
| `grad_clip` | 4.196 | 4.400 |
| `huber_beta` | 0.853 | 0.960 |

**Results — two independent runs:**

| Metric | Run A | Run B (committed) |
|--------|-------|-------------------|
| Test RMSE (per-axis, scaled) | 0.1874 | **0.1875** |
| Test type accuracy | 97.64% | **97.42%** |

**Key findings:** For the first time across all experiments, the aggregate RMSE meaningfully decreased — from ~0.191 to 0.1875. The plateau in Experiments 1 and 2 was indeed a data coverage ceiling. As Section 5 shows, however, this improvement was *entirely* a monopole improvement: the per-type breakdown reveals that dipole error did not move at all between Experiments 1–3. More data helped the well-posed half of the problem and left the ill-posed half exactly where it was.

### 4.4 Experiment 4 — Spherical Output Constraint

**Research question:** Section 5.3 of the Experiment 3 analysis identified a *regression-toward-center* failure mode — ambiguous predictions collapsing toward the geocenter. Experiment 4 asks whether constraining the regression head to emit only physically valid shell positions removes this failure mode and lowers position error.

**Design:** a single controlled run using Experiment 3 Run B's exact hyperparameters. The only change is replacing the final linear layer of the regression head with a `SphericalOutputLayer`. This layer emits four numbers: a 3-vector that is L2-normalized to a unit direction, and a fourth passed through a sigmoid to a shell-constrained radius (`SHELL_MIN + sigmoid · SHELL_RANGE`); their product is the Cartesian position, so every prediction is a valid shell point by construction. The direction-vector parametrization is symmetric in X/Y/Z and deliberately avoids the angular singularities of a latitude/longitude encoding (an earlier lat/lon version collapsed the Y-axis). The learning rate was raised from 1.48e-4 to 2.5e-4 to compensate for the gradient compression introduced by the sigmoid radius activation. All other hyperparameters, the dataset (64,000 samples), and the seeds are identical to Run B, so any change is attributable solely to the output constraint. Early stopping triggered at epoch 115; the model has 633,140 trainable parameters.

**Aggregate results (Exp 3 Run B vs. Exp 4):**

| Metric | Exp 3 (Cartesian) | Exp 4 (Spherical) |
|--------|-------------------|-------------------|
| Per-axis RMSE (scaled) | 0.1875 | 0.2416 |
| 3D RMSE (km) | 6,078 | 7,842 |
| Median error (km) | 7,117 | 3,152 |
| 90th percentile (km) | 9,035 | 13,790 |
| Type accuracy | 97.42% | 97.54% |

At the aggregate level the result looks contradictory — the median error more than halved while the RMSE and the 90th percentile got substantially worse. Section 5.3 resolves the paradox: these aggregate statistics are summarizing a *mixture* of two source types that the constraint affects in opposite directions. The constraint is best understood not as a "fix" but as a diagnostic intervention that cleanly separates the well-posed monopole sub-problem from the ill-posed dipole one.

---

## 5. Results

### 5.1 Cross-Experiment Comparison

To compare the four models on an identical footing, all four saved checkpoints were re-evaluated on **one common clean test set** (the seed-123 grid) in a single analysis notebook (Experiment 5). This *uniform re-evaluation* is the apples-to-apples comparison and is the source of the per-type results in Section 5.2. Its figures differ from each experiment's originally reported values for a concrete reason: Experiments 1 and 2 were trained and originally scored in the smaller 400-position regime, so re-scoring their checkpoints on this common, larger test set shifts their numbers (e.g. Exp 1 per-axis RMSE 0.1915 → 0.1971). For Experiments 3 and 4, which already used this test set, the only difference is minor GPU floating-point nondeterminism.

| Experiment | Type acc. | 3D RMSE (scaled) | 3D RMSE (km) | Median (km) | P90 (km) |
|------------|-----------|------------------|--------------|-------------|----------|
| Exp 1 (Broad search) | 97.64% | 0.3414 | 6,232 | 6,344 | 9,544 |
| Exp 2 (Loss/optim) | 96.99% | 0.3383 | 6,172 | 7,144 | 9,037 |
| Exp 3 (Cartesian) | 97.42% | 0.3249 | 6,078 | 7,117 | 9,035 |
| Exp 4 (Spherical) | 97.54% | 0.4187 | 7,842 | 3,152 | 13,790 |

The 3D Euclidean RMSE (≈ √3 × the per-axis value) is the physical position error reported in kilometres; the per-axis RMSE used in Sections 4.1–4.4 is the cross-experiment comparison metric inherited from the search objective.

### 5.2 The Decisive Result — Error by Source Type

Splitting the position error by source type is the single most informative view in this project. It explains every aggregate number in the table above.

| Experiment | Monopole median (km) | Monopole RMSE (km) | Dipole median (km) | Dipole RMSE (km) |
|------------|----------------------|--------------------|--------------------|------------------|
| Exp 1 (Broad search) | 1,238 | 2,305 | 8,450 | 8,506 |
| Exp 2 (Loss/optim) | 1,257 | 2,514 | 8,335 | 8,358 |
| Exp 3 (Cartesian) | **723** | 2,005 | 8,331 | 8,358 |
| Exp 4 (Spherical) | 722 | 1,647 | 10,977 | 10,967 |

Two facts dominate:

1. **Dipoles never improve.** Dipole median error is ~8,300–8,450 km in *every* Cartesian experiment, completely flat across the architecture search, the loss tuning, and the 8× data expansion. Dipole localization is not data-limited, capacity-limited, or optimizer-limited.
2. **Every aggregate "win" was a monopole win.** The headline Experiment 3 improvement (per-axis RMSE 0.191 → 0.1875) was monopoles improving from ~1,240 km to 723 km as independent spatial coverage grew. The well-posed sub-problem responded to more data exactly as expected; the ill-posed one did not.

### 5.3 Interpreting the Spherical Constraint

The per-type table also dissolves the Experiment 4 paradox.

**Monopoles were already well-localized, and the constraint barely touched them.** Monopole median error is 723 km under the Cartesian head (Exp 3) and 722 km under the spherical head (Exp 4) — essentially identical, ~8% of the mean shell radius. What the constraint *did* do for monopoles is trim the upper tail: monopole RMSE fell from 2,005 to 1,647 km and the 90th percentile from 1,880 to 1,580 km, because the spherical head cannot collapse the occasional ambiguous monopole toward the geocenter. The constraint did not unlock monopole localization — that was already solved by Experiment 3 — it only cleaned up a small number of hedged outliers.

**Dipoles got worse, and the reason is informative.** For an ambiguous dipole — one whose moment orientation cannot be recovered from the readings — the prediction that minimizes expected squared error is the *conditional mean* of the consistent positions, which lies in the interior, near the geocenter. Two closed-form anchors confirm this is exactly what the models do:

- **Cartesian dipoles ≈ geocenter prediction.** A prediction at the geocenter incurs an error equal to the source's geocentric radius. The mean shell radius is ≈ 8,322 km; the measured Cartesian dipole median is **8,331 km** — a 0.1% match. The Cartesian head is hedging ambiguous dipoles to the center, precisely the regression-toward-center behavior hypothesized in Experiment 3.
- **Spherical dipoles ≈ random shell points.** The mean separation of two independent points on the shell is ≈ 4R/3 ≈ 11,100 km. The measured spherical dipole error is **~11,000 km** (median 10,977, RMSE 10,967). Forced off the interior and onto the shell, the constrained model's dipole predictions fall on the scale of random shell points (~4R/3) — it extracts essentially no usable dipole position information.

So the spherical constraint forbids the RMSE-optimal interior hedge. For monopoles, where the hedge was rarely needed, this is harmless and even slightly helpful. For dipoles, where the hedge was doing real work to minimize squared error, removing it inflates dipole RMSE from 8,358 to 10,967 km. Because dipoles carry half the test set and the larger errors, the *aggregate* RMSE rises (6,078 → 7,842 km).

**The aggregate median is a mixture artifact — read it with care.** The aggregate median dropped from 7,117 to 3,152 km, which might look like a large improvement but is not a substantive one. With a two-population, bimodal error distribution (a tight monopole cluster near ~700 km and a broad dipole cluster near ~8,000–11,000 km), the overall median simply reports where the 50th percentile falls relative to the gap between the clusters. Under the Cartesian head the monopole upper tail overlapped the dipole cluster and pushed the 50th percentile up to ~7,100 km; under the spherical head the two populations separate cleanly, so the 50th percentile drops into the gap at ~3,150 km — even though the model became *worse* at the half of the data the statistic is partly measuring. The per-type table, not the aggregate median, is the honest summary.

**Net interpretation.** The residual position error after four experiments is *dominated by* dipole moment ambiguity, not model capacity, data volume, optimizer, or output parametrization. Monopole localization (~720 km) is good and improves with data; dipole localization (~8,300 km) is pinned to the geometric scale of the problem and does not respond to any lever tried here. Future progress on dipoles requires recovering the discarded moment information — not a better network.

### 5.4 Source-Type Classification

Classification is the easy task. Type accuracy is 97–98% in every experiment. In the committed Experiment 3 model the asymmetry is informative: monopole precision is 1.000 and dipole recall is 1.000, meaning every source labeled "monopole" truly is one, while ~5% of monopoles are misclassified as dipoles. The shallow single-layer classification head suffices because the 1/r² vs. 1/r³ falloff signatures are easy to separate even when the *position* is ambiguous.

---

## 6. Bayes Error Analysis

### 6.1 Motivation

Because all data was simulated with full knowledge of the data-generating process, it is possible to estimate the *scale* of the irreducible position ambiguity — the confusion any model would face because different sources can produce nearly identical sensor readings. This arises from two sources:

**Moment ambiguity.** A dipole's field depends on both position and moment orientation. Two sources at different positions with different orientations can produce nearly identical readings, so the best any model can do is predict the expected position averaged over all consistent configurations — which need not lie near any particular true position. (This is exactly the dipole floor measured directly in Section 5.2.)

**Sensor geometry.** The 29 stations are concentrated in certain regions. Sources over sparsely covered areas produce weaker, less spatially distinct field patterns, so their inverse problem is less constrained.

### 6.2 Centroid Nearest-Neighbor Proxy

To put a number on the ambiguity scale, a centroid nearest-neighbor proxy was computed. For each clean test sample, the nearest training *centroid* — the average of the 10 noise realizations of a (position, type) configuration, which approximates that configuration's clean field — is found in scaled input space, and the 3D gap between their true positions is recorded. (Averaging the realizations is essential: an earlier version that queried against raw *noisy* training inputs inflated the proxy to roughly the shell radius, because noise scatters every sample into its own neighborhood.)

**This proxy estimates the scale of irreducible ambiguity; it is not a strict lower bound.** Two effects bias it upward:

1. **Grid-spacing floor.** The test grid (seed 123) and training grid (seed 42) do not coincide, so the nearest training centroid sits at a genuinely different position even when the field is fully informative — the gap cannot fall below the grid resolution.
2. **Difference of two draws.** The proxy compares two samples from the set of positions consistent with a reading, rather than a sample against its conditional mean (the RMSE-optimal target). This inflates it by roughly a further √2.

A model can therefore legitimately score *below* the proxy. The shell-to-shell proxy is the right yardstick specifically for the shell-constrained model.

**Results:**

| Metric | Value |
|--------|-------|
| Proxy 3D RMSE (scaled / km) | 0.4245 / 7,942 km |
| Proxy median / P90 (km) | 2,907 / 14,017 km |

| Experiment | 3D RMSE (scaled) | vs. proxy | Reading |
|------------|------------------|-----------|---------|
| Exp 1 (Cartesian) | 0.3414 | −19.6% | below proxy via interior-mean hedge |
| Exp 2 (Cartesian) | 0.3383 | −20.3% | below proxy via interior-mean hedge |
| Exp 3 (Cartesian) | 0.3249 | −23.5% | below proxy via interior-mean hedge |
| Exp 4 (Spherical) | 0.4187 | −1.4% | **reaches proxy scale** |

The three Cartesian models sit ~20% below the proxy — not because they beat the irreducible floor, but because their RMSE-optimal predictor is the *interior* conditional mean, which a shell-to-shell distance cannot represent. The shell-constrained Experiment 4 model is the only one operating on the proxy's own manifold, and it lands essentially *on* the proxy (−1.4%). This is consistent with Experiment 4 being near the achievable ceiling for a shell-constrained predictor on this array — with the caveat that the proxy is itself an over-estimate (grid floor + difference-of-draws), so "reaches the proxy scale" means *near*, and above, the true Bayes floor rather than at it.

The per-type analysis of Section 5.2 makes the same point more directly than the proxy can: dipole error at the geocentric-radius scale (Cartesian) and the random-shell-point scale (spherical) is the irreducible ambiguity, expressed in closed form.

---

## 7. Discussion

### 7.1 What Worked

**Bayesian hyperparameter optimization** was far more efficient than grid search. Optuna identified the decisive parameters (activation, residual connections, batch size, classification-head depth) within the first ~10 trials of Experiment 1 and spent the rest refining the promising region. Total compute across all studies was a few hours on a T4 GPU.

**The multi-task architecture** performed well on both tasks simultaneously without evidence of task conflict, consistent with the shallow classification head and deeper regression head the search preferred.

**Dataset restructuring** from correlated noise realizations to independent spatial positions was the single most impactful change — but, as Section 5.2 clarifies, its benefit was confined to the well-posed monopole sub-problem.

### 7.2 What the Per-Type View Changed

The most important methodological lesson of this project is that an aggregate metric over a heterogeneous population can be actively misleading. The per-axis RMSE plateau looked like a single wall; it was actually two regimes — a solvable one and an unsolvable one — averaged together. Had the analysis stopped at aggregate RMSE, the natural (wrong) conclusion would have been "the model needs more capacity or data." The per-type split shows the opposite: capacity and data are irrelevant to the binding constraint, which is dipole moment ambiguity. The spherical-output experiment was valuable precisely because it failed in an informative way — its aggregate RMSE got worse, but it produced two closed-form confirmations (geocenter hedge for Cartesian dipoles, random-shell scatter for spherical dipoles) that pin down the cause.

### 7.3 Reproducibility and Notebook Hygiene

This project began life as a single monolithic notebook in which cells were run repeatedly and out of order during exploration. That workflow produced results that could not be regenerated from a clean top-to-bottom run: hidden state from earlier cells silently propagated into later ones, and "final" numbers depended on execution history rather than on the code as written. Diagnosing one of the early order-of-magnitude reporting errors traced directly back to this.

A substantial fraction of the effort here went into fixing that. The work was split into five self-contained notebooks (`01`–`05`), each re-including the simulator, sensor loader, and model definition so that any one runs standalone with **Runtime → Run all**. Random seeds were added at every stochastic step — data generation, the train/validation split, `torch.manual_seed`, and the Optuna sampler — so a clean run is reproducible. The cross-experiment comparison was deliberately rebuilt as a separate analysis notebook (`05`) that loads the saved checkpoints and re-evaluates them on one common test set, rather than trusting numbers carried over from each training session. The two-independent-runs benchmark in Experiment 3 (Section 4.3) is a direct product of this discipline. The lesson generalizes beyond this course: an experiment that cannot be reproduced from a clean state is not yet a result.

### 7.4 Limitations

The binding limitation is dipole moment ambiguity, quantified in Section 5.2. A feedforward network trained on (noisy field → position, type) pairs cannot recover moment information that was never in the training labels. A physics-informed architecture that explicitly models the forward field equations and inverts them — constraining predictions to be physically consistent — would be the natural next step, though it could only help where the forward problem is actually invertible. The sensor array's geographic coverage gaps (dense over Europe and North America, sparse over the Pacific, Southern Ocean, and poles) impose an additional, separate limit for sources over poorly covered regions.

---

## 8. Conclusion

This project demonstrated that a fully-connected multi-task neural network can classify ionospheric magnetic source type with 97–98% accuracy and localize sources with an aggregate 3D error of about 6,080 km — a number that, taken alone, is misleading. The decisive result is the decomposition by source type:

1. **Architecture matters less than expected.** Across 30 trials in Experiment 1, networks from 2 to 5 layers and 64 to 512 units produced nearly identical RMSE. Capacity is not the bottleneck at the scales tested.
2. **Loss-function tuning can be misleading.** Experiment 2 cut validation loss by 36% with no change in position error. Raw position error is the honest metric.
3. **Independent spatial coverage helps — but only the well-posed sub-problem.** The Experiment 3 data expansion improved monopole localization from ~1,240 km to ~720 km while leaving dipole error unchanged at ~8,300 km.
4. **The residual error is dominated by dipole moment ambiguity, not the model.** Monopoles are well-localized (~720 km) regardless of output head; dipoles sit at the geocentric-radius scale under a Cartesian head (8,331 km ≈ mean shell radius, the regression-toward-center hedge) and at the random-shell-point scale under a spherical head (~11,000 km ≈ 4R/3). No architecture, dataset, optimizer, or output parametrization tried here moved the dipole floor.

Future work should target the moment ambiguity directly — physics-informed inversion that models the forward equations, expanded sensor coverage to close geographic gaps, and richer labels (e.g., predicting moment orientation jointly) — rather than further scaling a model that is already at the physical floor for the half of the problem that is solvable, and at the ambiguity floor for the half that is not.

---

## Appendix A — Software and Hardware

| Component | Version / Specification |
|-----------|------------------------|
| Python | 3.12 |
| PyTorch | 2.4 |
| Optuna | 3.x |
| scikit-learn | Latest |
| Training hardware | NVIDIA T4 GPU (Google Colab) |
| Training time (total) | ~6–8 hours across all experiments |

## Appendix B — Hyperparameter Search Summary

| Parameter | Exp 1 Range | Exp 2 Range | Exp 3 Range | Exp 4 (fixed) |
|-----------|------------|------------|------------|---------------|
| `n_layers` | 2–5 | 3–5 | 3–5 | 3 (Run B) |
| `hidden_dim` | 64–512 | 288–448 | 288–512 | 448 |
| `dropout_rate` | 0.10–0.50 | 0.10–0.22 | 0.10–0.35 | 0.226 |
| `weight_decay` | 1e-6–1e-3 | 1e-4–1e-3 | 1e-4–1e-3 | 1.8e-4 |
| `learning_rate` | 1e-4–5e-3 | 1e-4–6e-4 | 1e-4–6e-4 | 2.5e-4 |
| `batch_size` | 32/64/128/256 | **128 (locked)** | **128 (locked)** | 128 |
| `activation` | relu/silu/gelu | **gelu (locked)** | **gelu (locked)** | gelu |
| `use_residual` | True/False | **False (locked)** | **False (locked)** | False |
| `n_type_layers` | 1–3 | **1 (locked)** | **1 (locked)** | 1 |
| `scheduler_factor` | — | 0.3/0.5/0.7 | **0.7 (locked)** | 0.7 |
| `huber_beta` | — | 0.01–1.00 | 0.70–1.00 | 0.960 |
| `grad_clip` | — | 0.50–5.00* | 2.0–6.0 | 4.400 |
| `bn_momentum` | 0.05–0.30 | 0.10–0.22 | 0.10–0.22 | 0.202 |
| **Output head** | linear | linear | linear | **spherical** |

*Exp 2 grad_clip tuning was ineffective due to a bug corrected in Exp 3.
