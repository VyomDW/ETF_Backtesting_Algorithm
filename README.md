# Pairs Trading with a Machine Learning Gatekeeper: Does It Actually Help?

## Summary

This project builds a statistical arbitrage (pairs trading) research pipeline —
from pair discovery through a machine learning "gatekeeper" model and a
realistic, cost-adjusted backtest — to answer a specific, falsifiable question:

**Does filtering pairs-trading signals with a machine learning classifier
improve risk-adjusted returns over a simple rules-based baseline, once
you account for realistic transaction costs and avoid look-ahead bias?**

The honest answer, after several rounds of increasingly rigorous testing:
**no, not convincingly.** The naive rules-based strategy consistently matched
or outperformed the ML-filtered versions on a risk-adjusted basis. That
negative result — and the process of rigorously trying to break an initially
promising-looking finding — is the actual contribution of this project.

---

## 1. Motivation

Pairs trading is a classic market-neutral strategy: find two assets that
historically move together, and when they temporarily diverge, bet on them
converging back. The hard part isn't finding correlated assets — it's knowing
whether a given divergence is a real trading opportunity (temporary noise) or
the start of a structural break (the relationship changing permanently).

This project asks whether a machine learning model, given features describing
the *shape* of a divergence (its magnitude, momentum, the pair's correlation
trend, market volatility context), can reliably tell these two cases apart —
and whether that ability translates into better trading outcomes once real
costs are applied.

## 2. Methodology

### 2.1 Pair discovery
Rather than hand-picking pairs, a universe of 25 liquid large-caps and sector
ETFs across 5 sectors (300 possible pair combinations) was screened for
cointegration using the Engle-Granger test. An early and important finding:
**a single 5-year cointegration test is unreliable** — the best-looking pair
initially (XLE/XLB) was only genuinely cointegrated in 13% of rolling 1-year
windows. This led to a rolling-window stability check becoming a standard
part of the pipeline, and a shift toward pooling events from many pairs
rather than betting on one "golden pair."

### 2.2 Dynamic hedge ratio and mean-reversion speed
Instead of a static OLS hedge ratio computed once over 5 years, a **Kalman
filter** estimates the hedge ratio (beta) fresh each day, letting it adapt as
the true relationship between two assets drifts. An **Ornstein-Uhlenbeck
process** is fit to each pair's spread to calculate its actual half-life of
mean reversion — replacing an arbitrary fixed holding period with a
data-derived one specific to each pair (ranging from 18 to 53 trading days
across the pool).

### 2.3 Event labeling
A "divergence event" is flagged when a pair's z-score crosses +/-2. Each
event is labeled based on what happens next, within a window of 1.5x that
pair's OU half-life:
- **Reverted** (success): z-score returns close to 0
- **Stopped out** (failure): z-score blows past a +/-3.5 stop-loss first
- **Timed out** (failure): neither happens within the window

Using a data-derived half-life window instead of an arbitrary 20-day guess
substantially changed the label distribution — under the old window, a
number of events that eventually reverted were being mislabeled as failures
simply because the model stopped watching too early.

### 2.4 Features
- Z-score magnitude, velocity (3-day, 5-day), and acceleration
- RSI ratio between the two legs
- Rolling correlation (20-day, 60-day) and its trend
- VIX level and 5-day slope (market regime context)
- The pair's own dynamic beta and OU half-life
- The pair's own historical reversion rate (computed causally, from prior
  events only)

### 2.5 Model validation
Standard random train/test splits were rejected early after producing
misleadingly optimistic results. The final validation uses **purged
walk-forward cross-validation**: training data is split chronologically into
expanding windows, and any training event whose outcome-evaluation window
overlaps with the test period is removed, closing a subtle information
leakage path.

Two models were compared: Random Forest and Logistic Regression. Random
Forest had higher raw accuracy, but **Logistic Regression had nearly 2x the
recall on actual failures** (53% vs 29%) — the economically important error
to avoid, since missing a real breakdown means staying in a losing position.
Logistic Regression was selected as the final gatekeeper for this reason,
not for its raw accuracy.

### 2.6 Backtest
The final backtest compares three approaches, all evaluated only on the
out-of-sample window from purged cross-validation (so the ML strategy is
never evaluated on data it trained on):
1. **Naive**: trade every divergence event
2. **Hard filter**: only trade events the classifier predicts will revert
3. **Probability-weighted**: trade every event, but size the position
   proportionally to the model's predicted confidence

Realistic costs are applied: 8 bps round-trip slippage per trade, plus a 1%
annualized short-borrow fee, prorated by how long each trade is actually
held (which matters significantly given holding windows of 20-150 days).

## 3. Results

**An important methodological note first:** an earlier, less rigorous version
of this backtest (static hedge ratio, arbitrary 20-day window, flat slippage
only, non-purged walk-forward validation) suggested the ML filter *did* add
value — better per-trade win rate and a better naive return/risk ratio than
the unfiltered baseline. That result did not survive tightening the
methodology: switching to a Kalman-filtered dynamic hedge ratio, an
OU-derived holding window, purged cross-validation, and realistic costs
(including a borrow fee scaled by holding period) reversed the finding. This
progression — an initially promising result failing to survive more rigorous
testing — is arguably the most important finding in this project.

| Strategy | Trades/yr | Win rate | Total return | Annualized return | Sharpe | Sortino | Max drawdown | Profit factor | Avg hold (days) | Turnover |
|---|---|---|---|---|---|---|---|---|---|---|
| Naive (trade every divergence) | 83 | 70.7% | 17.6% | 6.2% | 1.85 | 1.45 | -4.1% | 1.85 | 18.5 | 8.28x |
| ML-filtered (LogReg gatekeeper) | 49 | 72.9% | 9.2% | 3.3% | 1.07 | 0.78 | -4.6% | 1.63 | 19.4 | 4.89x |

The naive strategy outperforms the ML-filtered version on **every single
metric** in this table, not just Sharpe. The ML filter does very slightly
improve win rate (72.9% vs 70.7%), but that's the only dimension where it
wins — and it comes at the cost of nearly halving both trade frequency and
capital turnover, which drags down compounding enough to erase the benefit
everywhere else, including profit factor and Sortino (which isolates
downside risk specifically, so this isn't just a Sharpe quirk).

A supplementary experiment tested **probability-weighted position sizing**
(scaling each trade's size by the model's predicted confidence, instead of
a hard include/exclude filter) as an attempt to recover some of this lost
trade frequency while still using the model's signal:

| Strategy | Trades/yr | Win rate | Total return | Max drawdown |
|---|---|---|---|---|
| Naive (flat sizing) | 83 | 70.7% | 17.6% | -4.1% |
| Probability-weighted | 83 | 70.7% | 6.8% | -3.3% |

This preserved full trade frequency and modestly improved max drawdown, but
total return dropped even further than the hard filter — the model's
predicted probabilities are weak/noisy enough that they end up scaling most
trades down fairly uniformly rather than sharply distinguishing good trades
from bad ones, so risk and return both fall together rather than trading off
favorably.

## 4. Key findings

1. **A single cointegration test is not trustworthy** — pairs that look
   cointegrated over one static window are frequently unstable when tested
   on rolling windows. Continuous rescanning, not a fixed pair, is the
   realistic approach.
2. **Classification accuracy and economic value are not the same thing.**
   A model with mediocre overall accuracy can still be selectively useful
   (e.g., for catching failures) without that translating into better
   trading outcomes once trade frequency and cost effects are accounted for.
3. **The choice of validation window matters enormously.** Moving from an
   arbitrary 20-day labeling window to a data-derived ~53-day OU half-life
   window fundamentally changed the problem — and moving from a standard
   train/test split to purged walk-forward CV changed the apparent model
   quality again.
4. **A rigorously-tested "the ML doesn't help" result is more valuable than
   an untested "the ML helps" result.** The interesting contribution of this
   project isn't a winning strategy — it's the demonstration that an
   initially promising-looking result (the first backtest did favor the ML
   filter) did not survive more rigorous testing, and reporting that
   honestly.

## 5. Known limitations / future work

- **OU half-life leakage**: each pair's half-life was estimated once using
  its full 5-year spread history and applied uniformly to all events for
  that pair, including events early in the period. A fully rigorous version
  would re-estimate half-life on an expanding or rolling basis.
- **Sample size**: even pooled across 15 pairs, the final dataset (~370
  events, ~28 true "stopped out" failures) is small for training a
  classifier with 12 features; results should be read as directional, not
  as statistically robust in a strict sense.
- **No dynamic mid-trade re-evaluation**: positions are held until a fixed
  exit condition (revert, stop-loss, or timeout) is reached, rather than
  re-scoring the model's prediction daily and exiting early if confidence
  drops. This would add realism but requires point-in-time feature
  recomputation during open positions.
- **No market microstructure data**: order flow, bid-ask spread dynamics,
  and intraday execution effects are not modeled.
- **Programmatic pair discovery**: pairs were screened via brute-force
  pairwise testing across a hand-selected 25-ticker universe. A larger
  universe with unsupervised clustering (e.g., PCA + hierarchical
  clustering) before cointegration testing would reduce selection bias and
  scale better.

## 6. How to run this project

```bash
pip install yfinance pandas numpy statsmodels scikit-learn pykalman matplotlib
```

Run the pipeline scripts in order (each saves its output as a CSV/PNG that
the next script consumes):

1. `step2d_universe_screen.py` -- pull universe prices, screen all pairs for cointegration
2. `step3b_kalman_ou.py` -- Kalman filter + OU half-life demo (single pair)
3. `step4e_pooled_kalman_ou.py` -- generalize across the pooled pair universe
4. `step5d_purged_cv.py` -- train classifiers with purged walk-forward validation
5. `step6c_final_backtest.py` -- final backtest, naive vs. ML-filtered, with realistic costs
6. `step6d_probability_weighted.py` -- three-way comparison including probability-weighted sizing
7. `step6e_additional_metrics.py` -- Sortino ratio, profit factor, turnover, annualized return

## 7. Tech stack

Python, pandas, NumPy, statsmodels (cointegration, OLS, OU process),
scikit-learn (Random Forest, Logistic Regression), pykalman (dynamic hedge
ratio), yfinance (data), matplotlib (visualization).
