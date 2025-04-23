# 🧠 ParkinsonScorePrediction

> 🏆 4ᵗʰ place at Challenge Data ENS #159  
> 🎯 Goal: Reconstruct the neurologist-corrected OFF motor score in Parkinson’s patients by **modeling the noise** in real-world clinical observations  
> 🧪 Signal-based approach — we predict the **noise**, not the score itself

---

## 🧩 Context

- The OFF motor score is subjective and noisy by nature.
- Ground truth: expert-corrected scores derived from repeated observations.
- Major challenge: **massive missing values**, inconsistent clinical recordings, and highly heterogeneous patient profiles.

---

## 🔬 Methodology

### 📉 Advanced Imputation

- `on` and `off` values were imputed using supervised models (RandomForest, XGBoost, GradientBoosting).
- `age_at_diagnosis` imputed via **linear regression** (R² ≈ 0.97).
- Variables like `ledd` or `time_since_intake_xx` were dropped due to low signal-to-noise ratio and interpretability issues.

### 🎛️ Signal-Inspired Feature Engineering

Per-patient time-series processing to extract dynamic behavior:

- **Derivatives**: `diff_off`, `off_acceleration`, etc.
- **Smoothed trends**: `rolling_mean`, `trend_change`
- **Normalized deviations**: `off_deviation`, `relative_diff_off`
- **Volatility & stability**: `stability`, `volatility`
- **Trajectory metrics**: `trajectory_position`, `visit_progress`

> Goal: approximate a **temporal structure** underlying the “true” OFF motor score progression.

### 🔧 Noise Modeling

Instead of regressing directly on the target:
- We compute: `noise = target - off_observed`
- We train our models on the **noise** to learn systematic deviations.
- Final prediction:  
  \[
  \texttt{off_corrected} = \texttt{off_observed} + \texttt{model(noise)}
  \]

---

## 📁 Repository Structure

- `preprocess_data.py`: main pipeline for imputation + signal feature generation
- `final_test.py`: full pipeline for final model training with LightGBM
- `test.py`: patient-wise evaluation, quantile regression, visualizations
- `preprocess4.py`: modular validation-focused preprocessing
- `remplissageoff.ipynb`, `analysedonnees.ipynb`: prototyping notebooks
- `images pour slides/`: visuals for final report

---

## 🎯 Results

- ✅ Top 4 result on leaderboard (RMSE)
- 🚀 Significant boost in performance after signal-like features
- 🧠 Noise modeling provided better generalization than direct target prediction

---

## 📦 To Improve

- Repo restructuring:
  - Move core code to `/src/`
  - Place raw and interim datasets in `/data/`
  - Organize notebooks in `/notebooks/`
- Add a `train → eval → submit` script
- Per-module READMEs for better reproducibility

---

## 👩‍💻 Lead Contributor

**Amel**  
Developed the imputation engine, signal-inspired feature engineering, and noise modeling pipeline.  
🔗 [GitHub](https://github.com/encoreunpseudo)

---

## 📎 Reference

- [Challenge 159 – challengedata.ens.fr](https://challengedata.ens.fr/challenges/159)

