# Dynamic-ETF-Cointegration-Engine-with-Machine-Learning-Structural-Break-Detection
# Machine Learning–Enhanced Pairs Trading Strategy

A quantitative research project investigating whether machine learning can improve **statistical arbitrage / pairs trading** by predicting whether extreme divergences between historically related assets are likely to **mean-revert or break down**.

The project combines **cointegration analysis, statistical feature engineering, machine learning, and time-series validation** to build an end-to-end framework for identifying and evaluating potential pairs-trading opportunities.

---

## Overview

Pairs trading is a market-neutral strategy that attempts to profit when two historically related assets temporarily diverge from their long-run relationship.

The core research question of this project is:

> **Can machine learning distinguish between divergences that are likely to mean-revert and divergences that are likely to continue breaking down?**

Rather than predicting individual stock prices, the project focuses on the **relationship between two assets** and treats extreme deviations from that relationship as classification events.

The workflow evolved iteratively: initial sector-level pairs were tested for cointegration, unstable relationships were rejected, a broader universe was screened, and the most promising relationships were used to construct a machine-learning dataset.

---

## Research Pipeline

```text
Market Data
     │
     ▼
Pair Universe Construction
     │
     ▼
Cointegration Screening
     │
     ▼
Rolling Historical Stability
     │
     ▼
Spread & Feature Engineering
     │
     ▼
Divergence Event Detection
     │
     ▼
Reversion / Breakdown Labeling
     │
     ▼
Machine Learning Models
     │
     ▼
Chronological Validation
     │
     ▼
Walk-Forward Evaluation
```

---

## 1. Pair Selection

The project begins by testing economically related securities for long-run statistical relationships.

Initial candidates included sector ETFs and highly related companies such as:

* XLE / XLB
* XOM / CVX
* JPM / BAC
* HD / LOW
* KO / PEP
* XOP / XLE

The project then expanded to a broader universe of **25 liquid equities and sector ETFs**, producing:

**300 possible pair combinations**

Each pair was evaluated using the Engle-Granger cointegration test.

Pairs were initially screened using a one-year rolling window, followed by historical rolling-window analysis to determine whether the relationship remained stable over time.

This prevents selecting a pair simply because it happened to appear cointegrated during one isolated period.

---

## 2. Spread Construction

For a selected pair, the project models the relationship between the two assets using their log prices.

For example:

```text
Spread = log(JPM) - log(BAC)
```

A rolling 60-day mean and standard deviation are then used to calculate the spread's z-score:

```text
Z-score = (Spread - Rolling Mean) / Rolling Std
```

The z-score represents how far the current relationship has deviated from its recent historical level.

Large positive or negative values indicate an unusually large divergence.

---

## 3. Feature Engineering

The model uses features designed to capture both the magnitude of the divergence and the underlying market regime.

### Core Features

* Absolute entry z-score
* Individual asset volatility
* Rolling correlation
* Days since the spread crossed its rolling mean

### Extended Features

* 5-day z-score momentum
* 20-day correlation trend
* Relative volatility ratio
* Spread volatility trend
* Pair-specific historical reversion rate

These features attempt to distinguish a normal temporary divergence from a structural breakdown in the relationship.

---

## 4. Event Labeling

Instead of predicting daily stock returns, the project converts extreme divergences into discrete classification events.

An event occurs when:

```text
|z-score| > 2.0
```

The model then looks forward up to **20 trading days**.

An event is labeled:

### Reverted

if the spread returns to:

```text
|z-score| < 0.5
```

within the lookahead period.

### Broke Down

if the spread remains outside the reversion threshold throughout the evaluation window.

A cooldown period is also applied so that a single prolonged divergence is not incorrectly counted as multiple independent events.

---

## 5. Machine Learning

Two supervised learning models are evaluated:

### Logistic Regression

Provides an interpretable linear baseline for estimating the probability of mean reversion.

### Random Forest

Captures nonlinear relationships between divergence characteristics, volatility, correlation, and historical behavior.

Both models use class balancing to account for differences in the frequency of reversion and breakdown events.

---

## 6. Time-Series Validation

Financial data cannot be treated like randomly distributed observations.

Random train/test splitting can introduce look-ahead bias and produce overly optimistic results.

To address this, the project uses:

### Chronological Train/Test Split

Earlier events are used for training and later events are reserved for out-of-sample testing.

### Walk-Forward Validation

The training window expands through time while subsequent observations are repeatedly used as test sets.

```text
Train ───────► Test
Train ───────────► Test
Train ─────────────────► Test
Train ───────────────────────► Test
```

This more closely approximates how a model would actually be trained and deployed in a live trading environment.

Models are compared against:

* Majority-class baseline
* Random 50/50 baseline

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1 Score

---

## Key Research Insight

One of the most important findings during development was that **strong aggregate cointegration does not necessarily imply a stable trading relationship over time**.

For example, the initial XLE/XLB relationship appeared promising when evaluated over the full historical sample, but rolling analysis showed that it was cointegrated in only **13.4% of one-year rolling windows**.

This motivated the transition toward:

1. Broad pair screening
2. Rolling stability analysis
3. Event-based modeling
4. Walk-forward validation

Rather than assuming that a statistically significant historical relationship will remain reliable, the project explicitly tests whether the relationship persists across different market regimes.

---

## Project Structure

```text
ETF_Backtesting_Algortithm.py
README.md

Generated datasets / outputs:
├── sector_etf_prices.csv
├── granular_pair_prices.csv
├── universe_prices.csv
├── pair_screen_results.csv
├── top_candidates_stability.csv
├── jpm_bac_features.csv
├── pooled_labeled_events.csv
├── model_results.csv
└── walk_forward_results.csv
```

---

## Technologies

**Languages**

* Python

**Data & Analysis**

* Pandas
* NumPy

**Financial Data**

* yFinance

**Statistical Modeling**

* Statsmodels
* Engle-Granger Cointegration Test
* Rolling-window analysis

**Machine Learning**

* Scikit-learn
* Logistic Regression
* Random Forest

**Visualization**

* Matplotlib

---

## How to Run

Install the required dependencies:

```bash
pip install yfinance statsmodels pandas numpy scikit-learn matplotlib
```

Then run the Python script:

```bash
python ETF_Backtesting_Algortithm.py
```

The project downloads historical market data, performs pair screening, constructs the event dataset, trains the models, and generates evaluation outputs and visualizations.

> **Note:** The project was originally developed in Google Colab and can also be run cell-by-cell in a notebook environment.

---

## Limitations

This project is a quantitative research framework rather than a production trading system.

Current limitations include:

* Historical relationships may not persist in future market regimes.
* Cointegration tests can produce false discoveries when many pairs are screened.
* The current spread construction uses a simplified relationship between pair components.
* Classification performance does not necessarily translate directly into trading profitability.
* Transaction costs, slippage, and market impact are not yet fully incorporated into the research pipeline.
* Further validation is required before considering any strategy for real-world deployment.

---

## Future Improvements

Planned extensions include:

* Implementing a full transaction-cost-aware trading backtest
* Estimating dynamic hedge ratios
* Comparing ML-filtered trading signals against a traditional z-score strategy
* Adding Sharpe ratio, maximum drawdown, turnover, and profit-factor analysis
* Implementing purged walk-forward validation with an embargo period
* Applying multiple-testing corrections to large-scale pair screening
* Expanding the asset universe and incorporating liquidity constraints
* Testing model performance across different market regimes

---
