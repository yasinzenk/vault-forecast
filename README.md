
# 📊 Vault Return Forecast

This repository contains a reproducible pipeline for collecting, processing, and analyzing hourly time-series data from a DeFi vault, along with a lightweight modeling framework for short-horizon dynamics.

It covers:

- ⛓️ data extraction,
- 🧮 economically grounded feature engineering,
- 📈 exploratory time-series analysis,
- 🤖 predictive modeling.


## 1. 🔗 Data Collection

### 📡 **Data Sources**

All data is collected off-chain using the Morpho GraphQL API:

- vault-level historical state
- allocation snapshots
- underlying market-level metrics

### **🗂️ Generated Datasets**

1) **Vault hourly fundamentals** : *data/processed/vault_hourly.parquet*

Hourly time series (≥ 7 days) including:
- 📉 share_price (derived and scale-corrected)
- 💰 total_assets
- 🧾 total_supply
- 📊 apy, net_apy
- 🏦 tvl_usdc_proxy

⚠️ Important: share price derivation is explicitly scale-calibrated using the protocol’s canonical sharePriceNumber snapshot to avoid silent unit mismatches.

2) **Weighted Underlying Market Features** : *data/processed/market_weighted_features.parquet*

Economically meaningful features constructed as allocation-weighted aggregates of underlying markets:
- ⚙️ weighted_utilization
- 📈 weighted_supply_apy
- 📈 weighted_supply_apy

Weights are derived from a vault allocation snapshot (supplyAssetsUsd / totalAssetsUsd).

📌 For transparency and reproducibility, the allocation snapshot used is also stored: *data/processed/allocation_snapshot.parquet*


## 2. 🔍 Exploratory Data Analysis (EDA)

###  **🔑 Key Observations**

- 📈 Share price dynamics : smooth, strictly increasing behavior consistent with interest accrual; price levels are non-stationary, so analysis focuses on returns.
- 💰 Return profile : 24h returns are small, strictly positive, and carry-driven, with low volatility and limited tail risk.
- 🏦 Underlying markets : allocation-weighted utilization remains structurally high, indicating a capital-constrained vault; supply APY transmits market conditions to yield.
- 🔁 Autocorrelation & leakage : strong short-lag autocorrelation arises from overlapping 24h windows, creating artificial persistence and requiring strict time-based validation.
- 🔗 Feature relationships : weighted utilization is the most informative signal, while supply and borrow APYs are highly collinear.




## 3. 🧠 Model Proposal

- 🎯 **Problem framing** : modeled as a time-series regression predicting the 24h forward return. Evaluation prioritizes MAE over MSE to avoid over-penalizing rare volatility spikes; Huber loss is used to balance precision and robustness.
- 🧩 **Feature engineering** (context-aware design) : features are built to capture three temporal horizons:
    - Inertia (1h lag): captures short-term persistence in utilization.
    - Momentum (3h–6h lags): detects acceleration or deceleration in borrowing demand.
    - Trend & cycle (rolling 6h / 24h): smooths noise and contextualizes current conditions relative to the daily regime.
- 📏 **Model selection** (parsimony first) : given the small sample size, simple linear models are favored:
    - Ridge Regression handles severe multicollinearity across APY and lagged features.
    - Huber Regressor is selected as the final model for its robustness to liquidity-driven outliers while retaining interpretability.
- 🧪 **Validation & benchmarking** : performance is evaluated against a strong implied APY baseline (≈ SupplyAPY / 365), which represents the protocol’s theoretical forward return. Validation uses a strict chronological split with embargo to prevent leakage from overlapping return windows.
- 🚫 Why not complex models : high-variance non-linear models (e.g. XGBoost, LSTMs) are intentionally excluded due to high overfitting risk in a low-volatility, data-scarce setting.

### **Results & Performance Analysis**

We evaluated two predictive models (Ridge, Huber) against two distinct baselines over a held-out test set (29 hours). 

Despite the limited sample size (121 hours), the Huber Regressor demonstrated significant predictive power, outperforming all benchmarks.1. 

**Key Findings** 

- **The "implied APY"** : the theoretical baseline (APY / 365) performed significantly worse than a simple historical mean. This confirms that the instantaneous APY displayed by the protocol is  volatile and serves as a noisy estimator for realized 24h returns.
- **Ridge Conservatism**: The Ridge model essentially converged to the mean baseline. Due to the small sample size and high regularization, it suppressed coefficients to near-zero, failing to capture the underlying signal.
- **Huber Robustness:** The Huber model successfully extracted a predictive signal, reducing the MAE by 12.3%. Its lower regularization penalty allowed it to assign significant weight to features while its loss function remained robust to the liquidity spikes observed in the training data.


Feature Importance & Economic InterpretationThe Huber model's coefficients reveal the economic drivers of the vault's return:

1. **Dominant Signal:** weighted_utilization_roll_mean_24h (Coef: 0.000522) is by far the strongest predictor. This confirms that sustained utilization pressure over a full day, rather than instantaneous spikes, is the primary driver of share price appreciation.
2. **Volatility as a Predictor:** weighted_supply_apy_roll_std_24h (Coef: 0.000256) ranks second. The model identified that high volatility in APY often precedes regime shifts in returns, acting as a secondary risk/reward signal.
3. **Inertia vs. Noise:** instantaneous features (like raw apy or weighted_utilization) received low weights (0.000052), validating our feature engineering strategy: smoothed, rolling metrics are superior to raw snapshots for forecasting low-volatility yield bearing assets.


## ⚠️ Limitations & Production Considerations

###  **Data & Target Limitations**

- Overlapping target construction
The 24h forward return is computed using a rolling window, which introduces strong short-term autocorrelation. This creates artificial persistence and increases the risk of overly optimistic performance if validation is not handled carefully.

- Low signal-to-noise regime
The target variable is extremely stable (USDC carry yield), with very low variance. As a result, absolute error metrics can appear small even for weak models, making relative performance improvements difficult to interpret.

- Short historical window
The analysis is based on ~7 days of hourly data. This is sufficient for methodological validation, but insufficient to capture rare events, long-term regime shifts, or stress scenarios.

###  **Feature & Model Limitations**

- Feature redundancy
Supply and borrow APYs are nearly perfectly collinear. While included for completeness, they provide limited incremental signal beyond utilization and must be handled carefully in more complex models.

- Linear modeling assumptions
Ridge and Huber regressions assume mostly linear relationships. While appropriate for a first-pass baseline in a low-volatility environment, they may fail to capture nonlinear dynamics during liquidity shocks.

- Baseline competitiveness
Simple baselines (mean return, APY-implied return) already perform strongly. This indicates that much of the predictable signal is driven by carry mechanics rather than short-term market dynamics.

###  **Validation & Leakage Risks**


- Time-series leakage risk
Because the target depends on future prices, strict timestamp-based splits and embargo windows are required. Any deviation (random splits, index-based splits) would invalidate results.

- Backtest ≠ live performance
Even with correct splits, historical backtests cannot fully reflect live inference constraints (delays, missing data, sudden allocation changes).

###  **Production Considerations**

- Retraining strategy
In production, the model should be retrained on a rolling or expanding window to adapt to evolving utilization and rate regimes.

- Monitoring & drift detection
Key metrics to monitor:

    - Input feature distributions (utilization, APYs)
    - Prediction error drift (MAE/RMSE over time)
    - Sudden changes in allocation structure or liquidity usage

- Model fallback & robustness
Given the dominance of carry mechanics, a simple APY-based estimator should always be maintained as a fallback baseline in case of model degradation.
## ▶️ Installation

Create & activate the virtual environment (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the project + dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Data collection (fetch 7d hourly)

```bash
python -m scripts.fetch_vault_hourly
python -m scripts.fetch_weighted_market_features
```

Train the model

```bash
python -m scripts.train_model
```

Predict last 24h (inference)

```bash
python -m scripts.predict_last_24h
```

### **Data reproducibility (important)**

This repo’s data collection scripts query live protocol endpoints. If you re-run the fetch scripts later, you will get a different dataset (time-shifted by “now”), which means:

- The regenerated vault_hourly.parquet / market_weighted_features.parquet will not match the data used in the included EDA.

- Consequently, the EDA figures and the model metrics/results will likely not be identical to what was originally reported.

**How to reproduce the exact reported results**

To reproduce the reported EDA + model results, do not re-run the fetch scripts. Instead:

```bash
python -m scripts.train_model
python -m scripts.predict_last_24h
```

**If you want fresh results on fresh data**

```bash
python -m scripts.fetch_vault_hourly
python -m scripts.fetch_weighted_market_features
python -m scripts.train_model
python -m scripts.predict_last_24h
```

This will produce a fully consistent set of artifacts for the new time window, but it will not match the original EDA/model outputs.

## 🤖 AI Usage Disclosure

In compliance with the assignment guidelines, I utilized AI tools (LLMs) to assist with code refactoring, documentation structure, and debugging, while maintaining full ownership of the architectural design, economic reasoning, and data analysis.