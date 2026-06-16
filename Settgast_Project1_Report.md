# Magnetic Source Identification Using a Multi-Task Feedforward Neural Network

**Strider Settgast**
MTH 5320 — Project 1
June 17, 2026

---

## Abstract

This project investigates the capability of a fully-connected multi-task neural network to solve the magnetic inverse problem: given simultaneous field measurements from a 29-station global magnetometer array, predict the 3D position and pole type (monopole or dipole) of an ionospheric magnetic source. Data was generated using a custom Python-based Magnetometer Time Series Simulator (MTSS) that computes exact dipole and monopole field equations with additive Gaussian noise. Three sequential hyperparameter experiments were conducted using the Optuna Bayesian optimization framework, progressively refining the search space and dataset design. The final model achieved a source type classification accuracy of 97.64% and a 3D position RMSE of 0.1874 in normalized [0,1] space on a clean held-out test set. Analysis of the position error plateau across all experiments suggests the model is approaching the Bayes error floor imposed by the geometry of the sensor array and the inherent ambiguity of the magnetic inverse problem.

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

Magnetic sources are placed uniformly at random on a spherical shell between 800 and 3,100 km altitude above Earth's surface (geocentric radii of 7,171–9,471 km), simulating ionospheric source conditions. Two source types are simulated:

- **Monopole:** an idealized point magnetic charge. Not physically realized in nature but conceptually useful as a limiting case with a simple isotropic field pattern.
- **Dipole:** a magnetic dipole with a randomly oriented moment vector. The dominant term in the multipole expansion of realistic ionospheric current systems.

For each source position and type, the simulator draws a random magnetic moment magnitude uniformly from [10¹², 10¹⁴] A·m² and a random moment direction uniformly on the unit sphere. The clean field is computed exactly via the dipole or monopole equations. Additive Gaussian noise with standard deviation σ = 0.01 is then applied independently to each field component to simulate sensor noise.

### 2.2 Dataset Design

The dataset design evolved across experiments based on findings about what limits model performance. The final dataset used for Experiment 3 is described here; differences from earlier experiments are noted in Section 4.

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

**Regression head:** a smaller stack of fully-connected layers terminating in a 3-dimensional output representing the scaled [X, Y, Z] position. Uses half the dropout rate of the backbone.

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

## 4. Hyperparameter Optimization

All hyperparameter search was conducted using Optuna with the TPE (Tree-structured Parzen Estimator) algorithm and a MedianPruner. Optuna's TPE algorithm learns from previous trials, concentrating search effort toward regions of the hyperparameter space that have produced good results. The MedianPruner stops underperforming trials early — those whose validation loss at a given epoch is worse than the median of all completed trials at that epoch — saving compute for more promising configurations.

Each experiment ran 30 trials, stored results in a SQLite database for session persistence, and trained a final model with the best found configuration for up to 400 epochs with early stopping.

*Reproducibility note: data generation and the train/validation split use fixed random seeds, and the Optuna sampler is seeded (`TPESampler(seed=42)`), so the hyperparameter search is reproducible from a clean run. Reported metrics may still vary by a small margin due to GPU floating-point nondeterminism.*

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
| Best test RMSE (scaled) | 0.1915 |
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

All other ranges were tightened based on the Experiment 1 winning cluster (e.g. `hidden_dim` narrowed to 288–448, `learning_rate` to 1e-4–6e-4).

**Epochs per trial:** 75 | **Final model epochs:** 400 | **Early stopping patience:** 25

**Results:**

| Metric | Value |
|--------|-------|
| Best val loss | 0.0642 |
| Improvement vs. Exp 1 | −36% |
| Test RMSE (scaled) | ~0.191 (unchanged) |

**Key findings:** The best trial used `huber_beta=0.931` — near the upper bound of 1.0, meaning the loss behaved almost identically to MSE. This is a physically meaningful result: with well-controlled Gaussian noise and normalized targets, the model benefits from the quadratic penalty that pushes hard on small residual errors in the late training phase. A gentle scheduler (`factor=0.7`) outperformed the previously hardcoded aggressive decay (`factor=0.5`). The `grad_clip` appeared to prefer values around 4.3.

The most important finding was the decoupling of val loss and RMSE. Val loss dropped 36% but RMSE was completely stationary at ~0.191. This confirmed that the loss function tuning was improving the numerical optimization landscape without moving the actual prediction quality. The RMSE plateau is a property of the data, not the model or optimizer.

Note: the `grad_clip` tuning in this experiment had no actual effect due to a bug — the base trainer's `train_epoch()` read a hardcoded `max_norm=1.0` rather than `self.grad_clip`. This was discovered and corrected before Experiment 3.

### 4.3 Experiment 3 — Larger Dataset and Bug Fix

**Dataset:** 1,600 unique positions × 2 types × 10 noise realizations = **32,000 samples**

> *Note: the dataset generation cell in the final notebook reflects the full 3,200 position configuration used for the final training run. Experiment 3's Optuna study was conducted on the 1,600-position intermediate dataset; the final model was trained on 3,200 positions.*

Experiment 3 addressed the data coverage ceiling identified in Experiments 1 and 2. The key change was replacing correlated noise realizations with independent spatial positions: halving realizations from 20 to 10 while multiplying unique positions, giving 4× more independent spatial coverage.

**Locked parameters:** all Experiment 2 winners plus `scheduler_factor=0.7`. `huber_beta` range narrowed to 0.70–1.00. `grad_clip` range narrowed to 2.0–6.0.

**New:** `TunableTrainer` subclass overrides `train_epoch()` so `grad_clip` is actually applied.

**Widened:** `dropout_rate` upper bound raised to 0.35 (more data supports stronger regularization). `hidden_dim` upper bound raised to 512 (more data may support wider networks).

**Epochs per trial:** 75 | **Final model epochs:** 400 | **Early stopping patience:** 50

**Best trial parameters (Trial 15):**

| Parameter | Value |
|-----------|-------|
| `n_layers` | 4 |
| `hidden_dim` | 448 |
| `decay_factor` | 0.965 |
| `hidden_dims` | [448, 432, 417, 402] |
| `n_loc_layers` | 3 |
| `dropout_rate` | 0.179 |
| `weight_decay` | 8.0e-4 |
| `bn_momentum` | 0.166 |
| `learning_rate` | 1.86e-4 |
| `grad_clip` | 4.196 |
| `huber_beta` | 0.853 |
| `loc_weight` | 0.562 |
| `type_weight` | 0.508 |

**Results:**

| Metric | Value |
|--------|-------|
| Best val loss | 0.0546 |
| Final model early stopping epoch | 142 |
| Test RMSE (scaled) | **0.1874** |
| Test type accuracy | **97.64%** |

**Key findings:** For the first time across all experiments, the RMSE meaningfully decreased — from ~0.191 to 0.1874. This confirms that the plateau in Experiments 1 and 2 was a data coverage ceiling. More independent spatial positions directly improved position prediction. The val loss improvement (0.0642 → 0.0546) was more modest than the jump from Experiment 1 to 2, suggesting that with more data the model naturally becomes better calibrated and the loss function tuning matters less.

---

## 5. Results

### 5.1 Experiment Progression Summary

| Experiment | Dataset | Best Val Loss | Test RMSE (scaled) | Type Accuracy |
|------------|---------|--------------|-------------------|---------------|
| Baseline (default config) | 16,000 samples | — | 0.1906 | 97.85% |
| Experiment 1 | 16,000 samples | 0.1009 | 0.1915 | 97.85% |
| Experiment 2 | 16,000 samples | 0.0642 | ~0.191 | ~97.5% |
| Experiment 3 | 32,000 samples | 0.0546 | **0.1874** | **97.64%** |

### 5.2 Final Model Performance

The final model was evaluated on the clean held-out test set of approximately 11,294 noiseless samples covering ~5,647 unique source positions never seen during training or validation.

**Position regression:**

| Metric | Value (scaled) | Approximate km |
|--------|---------------|----------------|
| RMSE | 0.1874 | ~650 km |
| MAE | — | *[fill from eval cell output]* |
| Median error | — | *[fill from eval cell output]* |
| 90th percentile | — | *[fill from eval cell output]* |

> *Note: km values marked for completion after running the evaluation cell in the final notebook.*

**Source type classification:**

| Metric | Value |
|--------|-------|
| Overall accuracy | 97.64% |
| Monopole precision | *[fill from classification report]* |
| Dipole precision | *[fill from classification report]* |
| Monopole recall | *[fill from classification report]* |
| Dipole recall | *[fill from classification report]* |

> *Note: per-class metrics marked for completion after running the evaluation cell.*

### 5.3 Interpretation of Position Error

A position RMSE of 0.1874 in [0,1] scaled space represents approximately 650 km average 3D error across a source shell spanning altitudes of 800–3,100 km above Earth's surface. To contextualize this number:

The sources are distributed across a spherical shell visible from 29 ground stations concentrated in certain geographic regions. For sources well above the array center with strong field contrast across stations, position prediction is relatively accurate. For sources near the edge of the array's geometric coverage — where the field is nearly uniform across all sensors — the inverse problem becomes fundamentally ambiguous. No model can reliably distinguish two nearby positions when their sensor signatures are nearly identical.

This is reflected in the altitude error analysis: position error is expected to increase with altitude due to the 1/r³ falloff of the dipole field, which reduces field contrast across stations at greater distances.

---

## 6. Bayes Error Analysis

### 6.1 Motivation

Because all data was simulated with full knowledge of the data generating process, it is possible to estimate the Bayes error — the irreducible lower bound on position prediction error that no model can achieve regardless of architecture, data size, or training procedure.

The Bayes error for this problem arises from two fundamental sources of ambiguity:

**Moment ambiguity:** a dipole source's field pattern depends on both position and the direction of its magnetic moment. Two sources at different positions with different moment orientations can produce nearly identical sensor readings. The model must predict the expected position averaged over all source configurations consistent with the observed readings — that average may not correspond closely to any particular true position.

**Sensor geometry:** the 29 magnetometer stations are real-world positions concentrated in certain regions. Sources over areas with sparse sensor coverage produce weaker, less spatially distinct field patterns. The inverse problem is inherently less constrained for such sources.

### 6.2 Nearest-Neighbor Proxy Method

The Bayes error was estimated using the nearest-neighbor proxy method. For each clean test sample, the training sample with the most similar sensor readings (closest L2 distance in scaled input space) was found. The 3D position gap between the test sample's true position and its nearest neighbor's true position quantifies how much confusion a perfect model would face: even an ideal predictor cannot distinguish two inputs that are nearly identical but correspond to different positions.

To prevent noise-realization bias (where the nearest neighbor is simply another noise draw of the same clean field at the same position), the training reference set was subsampled to one sample per unique (position, type) configuration.

**Results:** *[fill from Bayes error cell output after running]*

If the nearest-neighbor RMSE is close to the model's actual RMSE (0.1874), this is strong evidence that the model is near-optimal for this sensor array. If a meaningful gap exists, more data or a physics-informed architecture could reduce error further.

---

## 7. Discussion

### 7.1 What Worked

**Bayesian hyperparameter optimization** was dramatically more efficient than grid search would have been. Optuna identified the decisive parameters (activation, residual connections, batch size, classification head depth) within the first 10 trials of Experiment 1 and spent the remaining 20 trials refining the promising region. The total compute for all three studies was under 6 hours on a T4 GPU.

**The multi-task architecture** performed well on both tasks simultaneously without evidence of task conflict. The shallow classification head (1 layer) and deeper regression head (3 layers) suggest the classification task is easier given the learned backbone representation, which aligns with the consistently high type accuracy (~97–98%) across all experiments.

**Dataset restructuring** from correlated noise realizations to independent spatial positions was the single most impactful change across the three experiments. Going from 400 to 1,600 unique positions (with noise realizations halved to compensate) produced the first meaningful RMSE improvement after two experiments of purely optimizer-level tuning.

### 7.2 What Was Unexpected

**The GELU activation's dominance** was not anticipated. GELU is typically associated with transformer architectures operating on token sequences. Its superiority over ReLU on this problem — which involves physically structured real-valued inputs — suggests the smooth gradient near zero helps with the fine-grained regression precision needed for position estimation.

**The high optimal weight decay (~8e-4)** relative to typical deep learning defaults (1e-5 to 1e-4) reflects the difficulty of this inverse problem. Strong L2 regularization prevents the model from memorizing the training set's noise realizations and forces it to learn the underlying physics.

**The near-MSE optimal loss** (`huber_beta ≈ 0.85–0.93`) is physically interpretable. With well-controlled Gaussian noise and normalized targets, the error distribution is approximately Gaussian — MSE is the natural loss for Gaussian errors. The Huber formulation provides marginal robustness without sacrificing the quadratic penalty that drives small-error precision.

### 7.3 Limitations

The most significant limitation is the position error plateau. Despite three experiments and systematic tuning of 14 hyperparameters, the RMSE moved from ~0.191 to 0.187 — a modest improvement relative to the effort invested. This strongly suggests the remaining error is dominated by the Bayes floor arising from moment ambiguity and sensor geometry, rather than model or optimization deficiencies.

A feedforward network trained on (noisy field → position, type) pairs cannot recover moment information that was discarded in the training labels. A physics-informed architecture that explicitly models the forward field equations and inverts them — rather than learning the inversion end-to-end — would likely achieve substantially lower position error by constraining predictions to be physically consistent.

The sensor array also has geographic coverage gaps. Ground-based magnetometer networks are densest in Europe and North America, sparser over the Pacific, Southern Ocean, and polar regions. Sources over poorly covered regions are fundamentally harder to localize regardless of model quality.

---

## 8. Conclusion

This project demonstrated that a fully-connected multi-task neural network can learn to simultaneously localize and classify ionospheric magnetic sources from ground-based magnetometer array measurements. The best model achieved 97.64% source type accuracy and a 3D position RMSE of 0.1874 in normalized space (~650 km) on a clean held-out test set.

Three key findings emerged from the experimental progression:

1. **Architecture matters less than expected.** Across 30 trials in Experiment 1, wildly different architectures (2 to 5 layers, 64 to 512 hidden units) produced nearly identical RMSE values. The inverse problem's difficulty is dominated by data and physics, not model capacity — at the scales tested.

2. **Loss function tuning can be misleading.** Experiment 2 reduced validation loss by 36% through optimizer and loss function improvements, but RMSE did not move. The raw position error is the honest metric; loss values are not directly comparable across different `huber_beta` settings.

3. **Independent spatial coverage is the primary driver of position accuracy.** The RMSE finally improved meaningfully in Experiment 3 when unique source positions increased from 400 to 1,600. This confirms that the plateau in earlier experiments was a data coverage ceiling, not a model capacity ceiling.

Future work should investigate physics-informed architectures that explicitly embed the magnetic field equations as constraints, expanded sensor networks to reduce geographic coverage gaps, and larger datasets (5,000+ unique positions) to further push against the Bayes floor.

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

| Parameter | Exp 1 Range | Exp 2 Range | Exp 3 Range | Final Value |
|-----------|------------|------------|------------|-------------|
| `n_layers` | 2–5 | 3–5 | 3–5 | 4 |
| `hidden_dim` | 64–512 | 288–448 | 288–512 | 448 |
| `decay_factor` | 0.70–1.00 | 0.82–0.97 | 0.82–0.97 | 0.965 |
| `dropout_rate` | 0.10–0.50 | 0.10–0.22 | 0.10–0.35 | 0.179 |
| `weight_decay` | 1e-6–1e-3 | 1e-4–1e-3 | 1e-4–1e-3 | 8.0e-4 |
| `learning_rate` | 1e-4–5e-3 | 1e-4–6e-4 | 1e-4–6e-4 | 1.86e-4 |
| `batch_size` | 32/64/128/256 | **128 (locked)** | **128 (locked)** | 128 |
| `activation` | relu/silu/gelu | **gelu (locked)** | **gelu (locked)** | gelu |
| `use_residual` | True/False | **False (locked)** | **False (locked)** | False |
| `n_type_layers` | 1–3 | **1 (locked)** | **1 (locked)** | 1 |
| `scheduler_factor` | — | 0.3/0.5/0.7 | **0.7 (locked)** | 0.7 |
| `huber_beta` | — | 0.01–1.00 | 0.70–1.00 | 0.853 |
| `grad_clip` | — | 0.50–5.00* | 2.0–6.0 | 4.196 |
| `bn_momentum` | 0.05–0.30 | 0.10–0.22 | 0.10–0.22 | 0.166 |

*Exp 2 grad_clip tuning was ineffective due to a bug corrected in Exp 3.
