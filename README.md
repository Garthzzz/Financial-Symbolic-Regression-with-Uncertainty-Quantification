# FSR-IE: Financial Symbolic Regression with Interval Estimation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Discovering Interpretable Financial Equations with Honest Uncertainty Quantification**

## Overview

FSR-IE is a symbolic regression framework designed specifically for financial data, addressing the fundamental challenge of equation discovery in low signal-to-noise ratio (SNR) environments. Unlike traditional symbolic regression methods that produce point estimates, FSR-IE outputs **interval-valued coefficients** that honestly reflect parameter uncertainty—wide intervals when signals are weak, narrow intervals when relationships are robust.

### Key Contributions

1. **SNR-Adaptive Inference**: Automatic detection of signal quality with domain-specific response strategies
2. **Interval-Valued Equations**: Bootstrap-based coefficient intervals that communicate uncertainty transparently
3. **Economic Constraint Integration**: Sign restrictions, bound constraints, and stationarity penalties grounded in financial theory
4. **Domain-Adjusted Scoring**: Fair evaluation across asset classes with inherently different predictability levels

## Motivation

Traditional symbolic regression methods face critical limitations in financial applications:

| Challenge | Traditional SR | FSR-IE Solution |
|-----------|---------------|-----------------|
| Low SNR (≈0.01–0.10) | Overfits noise | SNR-adaptive regularization |
| Parameter uncertainty | Point estimates only | Bootstrap confidence intervals |
| Economic validity | No constraints | Theory-driven sign/bound constraints |
| Cross-domain comparison | Single R² threshold | Domain-adjusted scoring |

Financial data is fundamentally different from physics benchmarks—market efficiency implies low predictability for returns (R² ≈ 0 is *correct*), while volatility exhibits moderate persistence (R² ≈ 0.3–0.5 is *excellent*). FSR-IE's scoring system reflects these domain realities.

## Method

### Algorithm Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FSR-IE Pipeline                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Stage 0    │    │   Stage 1    │    │   Stage 2    │          │
│  │  Diagnostics │───▶│  Selection   │───▶│  Estimation  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│        │                    │                    │                  │
│        ▼                    ▼                    ▼                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ SNR estimate │    │  Stability   │    │ Constrained  │          │
│  │ Noise level  │    │  selection   │    │    SLSQP     │          │
│  │ Denoising?   │    │  (bootstrap) │    │  Bootstrap CI│          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage 0: Data Diagnostics

SNR estimation via cross-validated R² with adaptive thresholds:

| SNR Level | Range | Response |
|-----------|-------|----------|
| High | > 0.50 | Standard regularization, narrow intervals expected |
| Medium | 0.15–0.50 | Moderate regularization, informative intervals |
| Low | < 0.15 | Strong regularization, denoising, wide intervals |

### Stage 1: Adaptive Stability Selection

Bootstrap-based feature selection with SNR-adaptive thresholds:

```python
# Selection probability via bootstrap Lasso
for b in range(n_bootstraps):
    idx = resample(n_samples)
    model = Lasso(alpha=α_snr).fit(X[idx], y[idx])
    selection_counts += (|coef| > ε)

selection_prob = selection_counts / n_bootstraps
selected = selection_prob ≥ threshold_snr  # threshold adapts to SNR
```

### Stage 2: Constrained Estimation with Intervals

**Constrained Optimization**: SLSQP with economic constraints
```python
minimize: MSE(y, ŷ) + λ||β||² + penalty_stationarity
subject to: sign_constraints, bound_constraints, sum_constraints
```

**Bootstrap Confidence Intervals**:
```python
for b in range(n_bootstraps):
    β_b = fit_constrained(X[resample], y[resample])
CI_95 = [percentile(β_boots, 2.5), percentile(β_boots, 97.5)]
```

### Interval Quality Classification

| Quality | Relative Width | Interpretation |
|---------|---------------|----------------|
| Precise | < 0.3 | Strong signal, reliable coefficient |
| Informative | 0.3–0.7 | Moderate uncertainty, usable with caveats |
| Wide | 0.7–1.5 | High uncertainty, interpret cautiously |
| Uninformative | > 1.5 | Signal indistinguishable from noise |

### Domain-Adjusted Scoring

Raw R² is misleading across financial domains. FSR-IE applies domain multipliers:

| Domain | Multiplier | Rationale |
|--------|------------|-----------|
| Volatility | 2.0× | Persistence creates moderate predictability |
| Returns | 5.0× | EMH implies very low SNR |
| Macro | 1.5× | Stable but regime-dependent |
| Curve Fitting | 1.0× | Deterministic relationships |
| Equilibrium | 1.5× | Factor structure recovery |

**Grade Thresholds** (on adjusted score):
- **A** (≥75): Publication-ready, robust relationship
- **B** (≥55): Strong evidence, recommended for use
- **C** (≥40): Moderate support, interpret with caution
- **D** (≥25): Weak evidence, consider alternatives
- **F** (<25): No meaningful signal detected

## Key Findings

### Validation Results

**Synthetic Data** (7 benchmark tests):

| Test | True Relationship | Test R² | Grade | Status |
|------|------------------|---------|-------|--------|
| Linear | y = 2 + 3x₁ - 1.5x₂ | 0.998 | A | ✓ Recovered |
| Polynomial | y = x² + 2xy + y² | 0.956 | A | ✓ Recovered |
| Trigonometric | y = sin(x) + cos(y) | 0.924 | A | ✓ Recovered |
| Exponential | y = exp(0.5x) | 0.987 | A | ✓ Recovered |
| High Noise (20%) | y = 2x + 3y + ε | 0.412 | B | ✓ Intervals widen appropriately |
| Multicollinear | Correlated features | 0.891 | A | ✓ Stable selection |
| Pure Noise | y = ε | -0.02 | F | ✓ Correctly rejected |

**Real Financial Data**:

| Application | Data Source | Period | Test R² | Adj R² | Grade |
|-------------|-------------|--------|---------|--------|-------|
| VIX Forecasting | CBOE | 2004–2024 | 0.368 | 0.736 | B |
| Factor Momentum | Kenneth French | 2000–2024 | -0.022 | 0.000 | F* |
| Crypto HAR | Yahoo BTC | 2018–2024 | -0.257 | 0.000 | F† |
| Okun's Law | FRED | 1959–2009 | 0.446 | 0.669 | B |
| Nelson-Siegel | Treasury | 2020–2024 | 0.944 | 0.944 | A |
| Fama-French | Simulated | — | 0.760 | 0.760 | A |

*\*Grade F for Factor Momentum is the **correct result**—validates Efficient Markets Hypothesis*  
*†Crypto HAR failure reflects weekly aggregation losing daily clustering signal*

### Discovered Equations (Real Data)

**VIX Mean-Reversion**:
```
VIX_{t+1} = 2.48 + [0.76, 0.98]·VIX_t + [0.63, 1.39]·ΔVIX_t − [−0.37, 0.13]·VIX_dev
```
- Strong persistence (β ≈ 0.87) with mean-reversion
- VIX deviation coefficient includes zero → uncertain contribution

**Okun's Law**:
```
ΔU = [0.30, 0.58] − [−0.54, −0.19]·ΔGDP
```
- Negative relationship confirmed (GDP growth reduces unemployment)
- Coefficient magnitude consistent with Okun's original estimate (≈−0.4)

## Installation

```bash
# Clone repository
git clone https://github.com/[username]/fsr-ie.git
cd fsr-ie

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

## Quick Start

The framework is implemented directly in the Jupyter notebooks. To use FSR-IE:

1. Open any notebook (e.g., `FSR_IE_Framework_v3.ipynb`)
2. Run Part 1 cells to define framework classes
3. Apply to your own data following the examples in Part 2

```python
# After running framework definition cells...

# Define economic constraints
constraints = ConstraintSpec(
    sign_constraints={0: 'positive'},      # First coefficient must be positive
    bound_constraints={0: (0, 1)},         # Bound between 0 and 1
    stationarity_penalty=0.1               # Penalize non-stationary solutions
)

# Initialize pipeline
fsr = FSRIntervalPipeline(
    constraint_spec=constraints,
    mandatory_features=[0],                # Always include first feature
    n_bootstraps=200,
    verbose=True
)

# Fit model
fsr.fit(X_train, y_train, feature_names=['x1', 'x2', 'x3'], 
        X_test=X_test, y_test=y_test)

# View results
fsr.print_report()

# Get domain-adjusted score
score = FinancialScorer.compute_score(
    train_r2=fsr.train_r2_,
    test_r2=fsr.test_r2_,
    domain='volatility',
    constraints_satisfied=fsr.report_.constraints_satisfied
)
print(f"Grade: {score['grade']}, Adjusted R²: {score['adjusted_r2']:.3f}")
```

## Repository Structure

```
fsr-ie/
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   ├── FSR_IE_Framework_v3.ipynb          # Synthetic data validation (7 tests)
│   ├── FSR_IE_RealData_Academic.ipynb     # Classic economic relationships (6 tests)
│   ├── FSR_IE_Research_v3.ipynb           # VRP, Factor Momentum, Crypto (synthetic)
│   └── FSR_IE_Research_RealData_v3.ipynb  # Same research with real data
└── data/
    └── README.md                           # Data sources documentation
```

> **Note**: The framework is currently implemented within notebooks. A standalone Python package (`pip install fsr-ie`) is planned for future release.

## Notebooks

| Notebook | Purpose | Data Type |
|----------|---------|-----------|
| `FSR_IE_Framework_v3.ipynb` | Algorithm validation on 7 benchmark tests | Synthetic |
| `FSR_IE_RealData_Academic.ipynb` | Classic relationships (Okun, Taylor, Phillips, Nelson-Siegel) | Real (FRED) |
| `FSR_IE_Research_v3.ipynb` | VRP, Factor Momentum, Crypto HAR | Synthetic |
| `FSR_IE_Research_RealData_v3.ipynb` | Same research questions with authentic data | Real (CBOE, French, Yahoo) |

## Limitations and Future Work

### Current Limitations

1. **Notebook-based implementation**: Framework classes are defined within notebooks; not yet packaged as standalone library
2. **Linear basis functions**: Current implementation discovers linear combinations; nonlinear transformations require manual feature engineering
3. **Univariate targets**: Extension to multivariate/system estimation not yet supported
4. **Static relationships**: Time-varying parameter models not yet implemented

### Planned Extensions

- [ ] **Package release**: Standalone `pip install fsr-ie` with modular architecture
- [ ] Integration with PySR for nonlinear symbolic search
- [ ] Bayesian posterior intervals (beyond bootstrap)
- [ ] Rolling window / regime-switching estimation
- [ ] GPU acceleration for large-scale applications

## Citation

If you use FSR-IE in your research, please cite:

```bibtex
@software{fsr_ie_2025,
  author = {[Author Name]},
  title = {FSR-IE: Financial Symbolic Regression with Interval Estimation},
  year = {2025},
  url = {https://github.com/[username]/fsr-ie}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kenneth French Data Library for factor returns data
- CBOE for VIX historical data
- Federal Reserve Economic Data (FRED) for macroeconomic series

---

**Contact**: [email] | **Issues**: [GitHub Issues](https://github.com/[username]/fsr-ie/issues)
