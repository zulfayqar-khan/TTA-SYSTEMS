# EXECUTION BLUEPRINT: Deep Learning Tactical Asset Allocation System

## For: Zulfiqar Khan — UFCEKP-30-3 Final Project
## Prepared by: Senior Quantitative AI Architect
## Purpose: Hand-off document for Claude AI to execute step-by-step

---

## PREAMBLE — CONTEXT FOR CLAUDE

You are building a production-grade Tactical Asset Allocation (TAA) system using deep learning. The system manages a portfolio of DJIA constituent stocks by dynamically adjusting portfolio weights in response to market conditions. The system must demonstrably outperform traditional portfolio strategies on risk-adjusted metrics across multiple market regimes. This is not a toy. Treat every design decision as if real capital depends on it.

**Universe:** Dow Jones Industrial Average (DJIA) constituents — approximately 30 stocks.
**Data Source:** Yahoo Finance (daily OHLCV + adjusted close).
**Output:** Daily or weekly portfolio weight vectors summing to 1.0 (long-only, fully invested).
**Evaluation:** Walk-forward backtest on held-out data, compared against multiple baselines.

---

## SECTION 0 — REFERENCE REPOSITORY ANALYSIS

**Repository:** `https://github.com/Musonda2day/Asset-Portfolio-Management-usingDeep-Reinforcement-Learning-`

Claude must study this repo for structural inspiration only. Here is what to borrow, what to reject, and what to redesign.

### 0.1 Ideas Worth Borrowing
- The framing of portfolio allocation as a sequential decision problem where the agent outputs weight vectors
- The use of a price-relative tensor (closing prices at t divided by closing prices at t-1) as a core input representation
- The concept of a portfolio vector memory that feeds back the previous allocation into the current decision
- The general structure: data → environment → agent → backtest → metrics

### 0.2 Parts That Are Insufficient for Production
- **Data handling:** The repo typically uses a small, fixed asset set with no survivorship bias correction, no handling of corporate actions, and no train/validation/test discipline suitable for time series
- **Feature engineering:** Relies almost entirely on raw price ratios; no macro features, no volatility features, no regime indicators
- **Training protocol:** No walk-forward validation; likely trains on one block and tests on another with no expanding/rolling window discipline
- **Transaction costs:** Either absent or naively implemented; no bid-ask spread modeling
- **Risk management:** No drawdown constraints, no volatility targeting, no portfolio concentration limits
- **Evaluation:** Typically reports cumulative return only; insufficient risk-adjusted analysis
- **Reproducibility:** Seeds, versioning, and deterministic training are usually absent

### 0.3 What Must Be Redesigned or Hardened
- The entire training/validation/test pipeline must use time-series-aware walk-forward splits
- Transaction costs must be integrated into both the training objective and the backtest
- The model must be evaluated under stress (2008 crisis, COVID crash, 2022 rate shock) not just average conditions
- Portfolio constraints (no shorting, max position size, turnover penalty) must be enforced at the architecture level, not post-hoc
- Reproducibility infrastructure (random seeds, config files, deterministic ops) must be built from day one

---

## SECTION 1 — SYSTEM-LEVEL ARCHITECTURE

Claude must build six distinct, modular components. Each is described below with its purpose and interface contract.

### 1.1 Data Pipeline

**Purpose:** Acquire, clean, validate, and store market data in a format ready for feature engineering. This component exists to ensure data integrity — garbage in, garbage out is the single most common failure mode in quantitative systems.

**Responsibilities:**
- Download daily OHLCV + adjusted close for all current and historical DJIA constituents from Yahoo Finance using `yfinance`
- Handle survivorship bias: include stocks that were removed from the DJIA during the study period, not just current constituents. Claude must research DJIA composition changes over the data window and include delisted/removed tickers. If historical composition data is unavailable, Claude must document this limitation explicitly and use the current 30 constituents with a clear caveat
- Detect and handle: missing data (forward-fill then back-fill, with a maximum gap of 5 trading days before dropping), stock splits (use adjusted close), dividends (use adjusted close), delistings
- Store as a multi-indexed Pandas DataFrame or HDF5 file with columns: Date, Ticker, Open, High, Low, Close, AdjClose, Volume
- Compute and store: daily log returns, daily simple returns, price-relative vectors (close_t / close_{t-1})
- **Data window:** Minimum 15 years (2008-01-01 to 2024-12-31). This captures the 2008 crisis, recovery, COVID crash, and 2022 rate hiking cycle

**Validation checks Claude must run:**
- No future data leakage: confirm all features at time t use only information available at or before time t
- No NaN values in the final processed dataset
- Return distributions: plot histograms and flag any returns exceeding ±20% daily (likely data errors)
- Correlation matrix of returns: flag any pair with correlation > 0.98 (redundant assets)

### 1.2 Feature Engineering Layer

**Purpose:** Transform raw market data into informative model inputs. This exists because raw prices are non-stationary and carry no predictive signal in themselves; the model needs derived features that capture momentum, volatility, mean-reversion, and cross-asset relationships.

**Feature groups (Claude must implement all):**

**Group A — Price-Based Features (per asset):**
- Log returns (1-day, 5-day, 21-day)
- Price relative to moving averages (close / SMA_20, close / SMA_50, close / SMA_200)
- Bollinger Band position: (close - lower_band) / (upper_band - lower_band)
- RSI (14-day)
- MACD signal line crossover (binary)

**Group B — Volatility Features (per asset):**
- Rolling realized volatility (21-day, 63-day)
- Garman-Klass volatility estimator (uses OHLC)
- Ratio of short-term to long-term volatility (21-day / 63-day) — a volatility regime indicator

**Group C — Cross-Sectional Features:**
- Cross-sectional rank of each asset's 21-day return (rank among all assets in the universe)
- Cross-sectional z-score of each asset's return
- Rolling pairwise correlation matrix (63-day window), compressed via PCA to top 5 principal components

**Group D — Market Regime Features (shared across all assets):**
- VIX level (download separately from Yahoo Finance: ^VIX) — if unavailable, substitute rolling 21-day volatility of SPY
- S&P 500 drawdown from rolling peak
- Yield curve slope proxy: use ^TNX (10-year yield) minus ^IRX (3-month yield) if available; otherwise omit with documented justification
- Market breadth: fraction of DJIA stocks with positive 21-day returns

**Normalization:**
- All features must be z-score normalized using a rolling window (252-day lookback). Claude must NOT use global normalization — this causes look-ahead bias
- After normalization, clip all features to [-3, +3] to prevent outlier distortion

**Input tensor shape:**
Claude must construct the final input as a 3D tensor: `(batch, lookback_window, num_assets × num_features)` or equivalently `(batch, lookback_window, num_assets, num_features)` if using a convolutional architecture.

**Lookback window:** 63 trading days (approximately 3 months). This is the default; it will be treated as a hyperparameter.

### 1.3 Model Layer

**Purpose:** Take the feature tensor at time t and output a portfolio weight vector w_t ∈ R^n where each w_i ≥ 0 and Σw_i = 1. This is the core intelligence of the system.

Full model strategy is detailed in Section 2.

### 1.4 Training & Validation Regime

**Purpose:** Train the model without overfitting and validate that it generalizes across time. This exists because financial data is non-stationary and has low signal-to-noise ratio; naive training produces models that memorize noise.

Full protocol is detailed in Section 3.

### 1.5 Backtesting Framework

**Purpose:** Simulate the model's portfolio performance on held-out data with realistic assumptions. This exists to provide a credible estimate of live performance.

**Requirements:**
- Execute trades at the next day's open price (not the close used to generate the signal — this prevents look-ahead bias)
- Apply transaction costs: 10 basis points (0.10%) per unit of turnover (|w_new - w_old|). This is conservative for liquid DJIA stocks
- Compute portfolio value daily: V_{t+1} = V_t × Σ(w_i × r_i) after costs
- Track: daily portfolio returns, cumulative returns, drawdown series, weight history over time
- Do NOT annualize metrics from periods shorter than 1 year

**Constraints enforced during backtest:**
- Long-only: all weights ≥ 0
- Fully invested: weights sum to 1.0 at all times
- Maximum single-asset weight: 30% (prevents dangerous concentration)
- Minimum rebalance threshold: only rebalance if Σ|w_new - w_old| > 5% (prevents excessive churning)

### 1.6 Evaluation & Benchmarking Layer

**Purpose:** Compare the model against baselines using rigorous metrics. This exists because "the model made money" is not sufficient — it must make money better than simple alternatives on a risk-adjusted basis.

Full specification in Section 4.

---

## SECTION 2 — MODEL STRATEGY

### 2.1 Primary Model: Temporal Convolutional Network (TCN) with Attention

**Why TCN over LSTM:**
- TCNs have stable gradients (no vanishing gradient problem), train faster via parallelization, and have been shown to match or exceed LSTMs on financial time series
- The dilated causal convolution structure naturally respects temporal ordering (no future leakage by architecture)
- Attention mechanism allows the model to learn which time steps and which assets are most relevant for the current allocation decision

**Architecture specification:**

**Input:** Tensor of shape `(batch, lookback=63, num_assets × num_features)`

**TCN Block:**
- 4 layers of dilated causal 1D convolutions with dilation factors [1, 2, 4, 8]
- Kernel size: 3
- Hidden channels: 64
- Residual connections between layers
- Dropout: 0.2 between layers (spatial dropout)
- Activation: GELU

**Attention Block:**
- Single-head self-attention over the temporal dimension applied to the TCN output
- This produces an attention-weighted summary of the lookback window

**Portfolio Head:**
- Flatten the attention-weighted representation
- Dense layer → 128 units → GELU → Dropout(0.3)
- Dense layer → num_assets units
- **Softmax activation** — this enforces w_i ≥ 0 and Σw_i = 1 by construction

**Output:** Weight vector w_t ∈ R^n

### 2.2 Secondary Model (for comparison): LSTM-based Allocator

**Why include this:**
- Provides a direct comparison to validate whether the TCN architecture adds value
- LSTMs are the most commonly used deep learning architecture in finance; if the TCN cannot beat an LSTM, the architectural choice is not justified

**Architecture:**
- 2-layer LSTM, hidden size 128, dropout 0.2
- Final hidden state → Dense(128) → GELU → Dropout(0.3) → Dense(num_assets) → Softmax

### 2.3 Optional Third Model: Deep Reinforcement Learning (PPO)

**Justification for RL:**
RL is appropriate here because TAA is inherently a sequential decision problem where today's allocation affects tomorrow's transaction costs and risk exposure. Supervised learning requires a "correct" label (target weight vector) which does not naturally exist; RL learns directly from reward signals (portfolio returns minus costs).

**Use only if the supervised models (TCN, LSTM) fail to outperform baselines.** RL is harder to train, harder to debug, and more prone to instability. It should be a fallback, not the first approach.

**If deployed:**
- Algorithm: Proximal Policy Optimization (PPO) — stable, well-understood, less hyperparameter sensitive than DDPG or SAC
- State: same feature tensor as above
- Action: portfolio weight vector (continuous action space, num_assets dimensions)
- Reward: daily log portfolio return minus transaction cost penalty minus a volatility penalty term (λ × rolling_volatility)
- Use the Stable-Baselines3 library for implementation reliability

### 2.4 Target Formulation and Loss Functions

**For the supervised models (TCN, LSTM):**

The target is NOT a "correct" weight vector (this does not exist). Instead, use a **differentiable portfolio return objective** — the model is trained end-to-end to maximize risk-adjusted portfolio performance.

**Primary loss function — Negative Sharpe Ratio:**
```
L = -mean(r_portfolio) / std(r_portfolio)
```
Where r_portfolio = Σ(w_i × r_i) computed over a training window (e.g., 21 days).

**Augmented loss (recommended):**
```
L = -SharpeRatio + λ_turnover × MeanTurnover + λ_concentration × MaxWeight
```
Where:
- MeanTurnover = mean(Σ|w_t - w_{t-1}|) penalizes excessive trading
- MaxWeight = mean(max(w_t)) penalizes concentration
- λ_turnover = 0.1, λ_concentration = 0.05 (starting values; tune as hyperparameters)

This is critical: the loss function must be aligned with the financial objective. Cross-entropy or MSE against arbitrary target weights is meaningless and will produce a system that optimizes the wrong thing.

---

## SECTION 3 — TRAINING & VALIDATION PROTOCOL

### 3.1 Data Split Strategy — Expanding Window Walk-Forward

**Claude must NOT use a single train/test split.** Financial data is non-stationary; a model that works in one period may fail in another. Walk-forward validation is the minimum acceptable standard.

**Split structure:**

| Phase | Period | Purpose |
|-------|--------|---------|
| Initial Training | 2008-01-01 to 2015-12-31 | First training window |
| Validation | 2016-01-01 to 2017-12-31 | Hyperparameter tuning, model selection |
| Test Fold 1 | 2018-01-01 to 2019-12-31 | Out-of-sample evaluation |
| Test Fold 2 | 2020-01-01 to 2021-12-31 | Out-of-sample (includes COVID) |
| Test Fold 3 | 2022-01-01 to 2024-12-31 | Out-of-sample (includes rate shock) |

**Walk-forward procedure:**
1. Train on 2008–2015, validate on 2016–2017, select best hyperparameters
2. Retrain on 2008–2017 with selected hyperparameters, test on 2018–2019
3. Retrain on 2008–2019, test on 2020–2021
4. Retrain on 2008–2021, test on 2022–2024
5. Report metrics on EACH test fold separately AND on the concatenated test period

**Embargo period:** Insert a 21-trading-day gap between the end of training data and the start of test data in each fold. This prevents information leakage from overlapping features (e.g., 21-day rolling averages computed using data that overlaps with the test period).

### 3.2 Hyperparameter Discipline

Claude must tune the following hyperparameters using the validation set ONLY:

| Hyperparameter | Search Range | Method |
|----------------|-------------|--------|
| Learning rate | [1e-5, 1e-2] | Log-uniform random search |
| Lookback window | {21, 42, 63, 126} | Grid |
| TCN hidden channels | {32, 64, 128} | Grid |
| Dropout rate | {0.1, 0.2, 0.3, 0.4} | Grid |
| λ_turnover | {0.01, 0.05, 0.1, 0.2} | Grid |
| λ_concentration | {0.01, 0.05, 0.1} | Grid |
| Batch size | {32, 64, 128} | Grid |

**Total budget:** Maximum 50 random configurations. Use Optuna or manual random search. Do NOT use exhaustive grid search — it is computationally wasteful and prone to overfitting the validation set.

**Early stopping:** Monitor validation Sharpe ratio. Stop training if no improvement for 20 epochs. Maximum 200 epochs.

### 3.3 Stability and Robustness Checks

Claude must perform ALL of the following after selecting the final model:

**Check 1 — Seed Sensitivity:**
Train the final model configuration with 5 different random seeds. Report the mean and standard deviation of all key metrics across seeds. If the standard deviation of the test Sharpe ratio across seeds exceeds 0.3, the model is unstable — flag this as a critical issue.

**Check 2 — Feature Ablation:**
Remove each feature group (A, B, C, D as defined in Section 1.2) one at a time and retrain. Report the change in validation Sharpe. This reveals which features are actually contributing vs. adding noise.

**Check 3 — Temporal Stability:**
Compute the rolling 6-month Sharpe ratio on the test set. Plot it. If the model's Sharpe drops below 0.0 for more than 3 consecutive months, investigate whether this coincides with a regime change and document findings.

**Check 4 — Weight Behavior Analysis:**
Plot the model's weight allocations over time. Flag if:
- The model consistently holds >80% in one asset (degenerate solution)
- Weights oscillate wildly from day to day (unstable signal)
- Weights are nearly equal across all assets (model has learned nothing beyond equal-weight)

### 3.4 Stress Testing

Claude must evaluate the trained model specifically on these crisis periods (which should be part of the test folds):

| Crisis | Period | What to Check |
|--------|--------|--------------|
| COVID crash | Feb 2020 – Apr 2020 | Max drawdown, did it reduce equity exposure? |
| 2022 rate shock | Jan 2022 – Oct 2022 | Sustained bear market behavior |
| 2018 Q4 selloff | Oct 2018 – Dec 2018 | Short sharp correction |

Report metrics for each sub-period separately. A model that performs well on average but collapses during crises is not deployment-ready.

---

## SECTION 4 — PERFORMANCE BENCHMARKS

### 4.1 Baseline Strategies Claude Must Implement

**Baseline 1 — Equal Weight (1/N):**
Allocate 1/N to each of the N assets. Rebalance monthly. This is the hardest naive baseline to beat and is the primary benchmark.

**Baseline 2 — Mean-Variance Optimization (Markowitz):**
Use a 252-day rolling window to estimate expected returns (sample mean) and covariance. Solve for the maximum Sharpe ratio portfolio with the same constraints (long-only, max 30% per asset). Rebalance monthly. Apply the same transaction costs.

**Baseline 3 — Momentum Strategy:**
Each month, rank assets by trailing 12-month return (excluding the most recent month). Go overweight the top quartile, underweight the bottom quartile. Specifically: top quartile gets 2/N weight each, bottom quartile gets 0.5/N weight each, middle gets 1/N. Normalize to sum to 1.

**Baseline 4 — Buy and Hold:**
Invest equal weight at the start of the test period and never rebalance.

**Baseline 5 — Risk Parity:**
Allocate inversely proportional to each asset's 63-day rolling volatility. Normalize to sum to 1. Rebalance monthly.

### 4.2 Metrics — Primary

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| Annualized Sharpe Ratio | (mean_daily_return × 252) / (std_daily_return × √252) | Primary risk-adjusted performance measure |
| Annualized Return | (1 + mean_daily_return)^252 - 1 | Raw performance |
| Annualized Volatility | std_daily_return × √252 | Risk level |
| Maximum Drawdown | Largest peak-to-trough decline | Worst-case loss |
| Calmar Ratio | Annualized Return / |Max Drawdown| | Return per unit of tail risk |

### 4.3 Metrics — Secondary

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| Sortino Ratio | Annualized return / downside deviation | Penalizes only bad volatility |
| Average Monthly Turnover | Mean of Σ|w_t - w_{t-1}| per month | Trading cost proxy |
| Hit Rate | % of months with positive return | Consistency |
| Profit Factor | Gross profit / gross loss (monthly) | Quality of winning vs losing |
| 95% CVaR (monthly) | Average of worst 5% monthly returns | Tail risk |

### 4.4 Minimum Performance Thresholds — Go/No-Go Criteria

The model is considered successful ONLY if ALL of the following hold on the concatenated out-of-sample test period:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Sharpe Ratio > Equal Weight Sharpe | By at least +0.15 | Must meaningfully beat the hardest naive baseline |
| Sharpe Ratio (absolute) | ≥ 0.5 | Below this, the model is not generating economically meaningful alpha |
| Maximum Drawdown | ≤ 35% | Must not expose the portfolio to catastrophic losses |
| Monthly Turnover | ≤ 60% | Must not churn the portfolio excessively |
| Sharpe > 0 in at least 2 of 3 test folds | — | Must not be regime-dependent to a single period |
| Seed stability | Sharpe std across 5 seeds < 0.3 | Must be reproducible |

If any criterion fails, the model does NOT qualify as successful. Claude must document which criteria failed and why.

---

## SECTION 5 — ANTI-FAILURE SAFEGUARDS

Claude must implement the following checks as explicit code functions that run automatically and raise warnings or errors.

### 5.1 Data Leakage Detection

**Check:** After constructing the feature matrix, verify that every feature at time index t is computed using ONLY data from indices ≤ t. Claude must write a unit test that:
1. Takes the feature computation function
2. Runs it on the full dataset
3. Runs it on a truncated dataset (first 80% of data only)
4. Verifies that the features for all dates within the truncated period are IDENTICAL in both runs
5. If they differ, there is look-ahead bias — HALT and fix

**Check:** Verify that the model never sees test-period returns during training. Print the date range of training data and test data for each fold and assert there is no overlap.

**Check:** Verify that normalization statistics (mean, std) used for features at time t are computed from a lookback window ending at t, not from the full dataset.

### 5.2 Overfitting Detection

**Check:** Compare training loss/Sharpe to validation loss/Sharpe. If training Sharpe > 3.0 while validation Sharpe < 0.5, the model is severely overfit. Reduce model capacity, increase dropout, or add regularization.

**Check:** Plot learning curves (training and validation performance vs. epoch). If training performance continuously improves while validation flatlines or degrades after epoch X, training should have stopped at epoch X.

**Check:** If the model's test performance is more than 50% worse than its validation performance (e.g., validation Sharpe = 1.2 but test Sharpe = 0.4), investigate whether the validation set is unrepresentative or the model has overfit to the validation period.

### 5.3 Regime Dependency Detection

**Check:** Compute test metrics separately for:
- Bull markets (months where S&P 500 return > +2%)
- Bear markets (months where S&P 500 return < -2%)
- Sideways markets (everything else)

If the model's Sharpe is positive in bull markets but negative in bear markets, it has likely learned a long-only momentum bias and is not truly tactical. Document this finding.

### 5.4 Economic Intuition Validation

**Check:** During the COVID crash (March 2020), did the model:
- Reduce overall equity exposure (i.e., shift toward lower-beta stocks)? If it increased exposure to the most volatile stocks during a crash, the model lacks economic sense.

**Check:** Examine the model's top holdings over time. Do they rotate, or does the model lock onto 2-3 stocks permanently? Permanent concentration suggests the model learned a stock-picking bias, not a TAA strategy.

**Check:** If the model's weight vector is nearly identical to the equal-weight vector (cosine similarity > 0.95 on average), the model has collapsed to the trivial solution and must be rejected.

### 5.5 Model Rejection Protocol

Claude must REJECT a model and document the reason if ANY of the following are true:
- Test Sharpe is negative on 2+ of 3 test folds
- Maximum drawdown exceeds 45%
- The model's weight vector has cosine similarity > 0.95 with equal weight on average
- The model's weight vector has zero variance across time (constant allocation)
- Training Sharpe exceeds 4.0 (almost certainly overfit)
- Seed sensitivity check shows Sharpe std > 0.5 across 5 seeds

---

## SECTION 6 — DELIVERABLES FROM CLAUDE (MANDATORY)

Claude must produce all of the following, in order. Each deliverable has an associated quality gate.

### 6.1 Step-by-Step Build Sequence

Claude must execute the build in this exact order:

**Phase 1 — Infrastructure (Estimated: 1 session)**
1. Set up project directory structure
2. Create a `config.yaml` with all hyperparameters, file paths, random seeds
3. Set global random seeds (Python, NumPy, PyTorch)
4. Install and verify all dependencies

**Phase 2 — Data (Estimated: 1-2 sessions)**
5. Build data download and cleaning pipeline
6. Run data validation checks (Section 1.1)
7. Save cleaned data to disk

**→ GATE 1: Data quality confirmed. No NaNs, no obvious errors, correct date ranges.**

**Phase 3 — Features (Estimated: 1-2 sessions)**
8. Implement all feature groups (A, B, C, D)
9. Implement rolling normalization
10. Construct input tensors
11. Run leakage detection unit test (Section 5.1)

**→ GATE 2: Feature leakage test passes. Feature distributions are reasonable (no extreme values).**

**Phase 4 — Baselines (Estimated: 1 session)**
12. Implement all 5 baseline strategies
13. Run backtests for all baselines on the full test period
14. Record baseline metrics — these are the targets to beat

**→ GATE 3: Baselines are implemented and producing sensible results. Equal-weight Sharpe is in a plausible range (typically 0.3–0.8 for DJIA).**

**Phase 5 — Model Build (Estimated: 2-3 sessions)**
15. Implement TCN + Attention architecture
16. Implement LSTM baseline architecture
17. Implement the differentiable Sharpe loss function
18. Implement training loop with early stopping
19. Implement walk-forward validation pipeline

**→ GATE 4: Models train without errors. Loss decreases over epochs. Weights are valid (sum to 1, all non-negative).**

**Phase 6 — Hyperparameter Tuning (Estimated: 1-2 sessions)**
20. Run hyperparameter search on validation set
21. Select best configuration for each model
22. Document selected hyperparameters and validation performance

**→ GATE 5: Best validation Sharpe is positive and exceeds equal-weight. If not, revisit feature engineering or model architecture before proceeding.**

**Phase 7 — Final Evaluation (Estimated: 1-2 sessions)**
23. Retrain final models using walk-forward protocol on each test fold
24. Run full backtest with transaction costs
25. Compute all primary and secondary metrics
26. Run all anti-failure safeguards (Section 5)
27. Run stress tests (Section 3.4)
28. Run stability checks (Section 3.3)

**→ GATE 6: Model passes the go/no-go criteria in Section 4.4. If not, document which criteria failed and whether the model still provides academic value even if not production-ready.**

**Phase 8 — Documentation & Deliverables (Estimated: 1 session)**
29. Generate all comparison tables, charts, and diagnostics
30. Write deployment-readiness checklist
31. Package all code, configs, and results

### 6.2 Decision Checkpoints (Go / No-Go Gates)

At each gate listed above, Claude must:
1. Print a summary of current status
2. List any anomalies or warnings
3. Explicitly state "GATE X: PASS" or "GATE X: FAIL — [reason]"
4. If FAIL, propose corrective action before proceeding

Claude must NEVER skip a gate. If a gate fails and cannot be remedied, Claude must document the failure and continue with explicit caveats on all downstream results.

### 6.3 Model Comparison Table

Claude must produce a final comparison table in this exact format:

| Metric | Equal Wt | Mean-Var | Momentum | Buy&Hold | Risk Parity | LSTM | TCN+Attn | (RL if used) |
|--------|----------|----------|----------|----------|-------------|------|----------|-------------|
| Ann. Return | | | | | | | | |
| Ann. Volatility | | | | | | | | |
| Sharpe Ratio | | | | | | | | |
| Max Drawdown | | | | | | | | |
| Calmar Ratio | | | | | | | | |
| Sortino Ratio | | | | | | | | |
| Avg Monthly Turnover | | | | | | | | |
| Hit Rate | | | | | | | | |
| 95% CVaR | | | | | | | | |

This table must be produced for:
1. Each test fold separately
2. The concatenated test period
3. Each stress sub-period

### 6.4 Diagnostics and Interpretation Guidance

Claude must produce the following visualizations:

1. **Cumulative return plot:** All strategies on one chart, log scale, with shaded recession/crisis periods
2. **Drawdown plot:** Underwater chart for the deep learning model vs. equal weight
3. **Weight allocation over time:** Stacked area chart showing how the model allocates across assets
4. **Rolling 6-month Sharpe:** Line chart comparing the model vs. equal weight over time
5. **Feature importance:** Bar chart from ablation study showing Sharpe impact of removing each feature group
6. **Learning curves:** Training vs. validation loss/Sharpe per epoch
7. **Turnover over time:** Bar chart of monthly portfolio turnover
8. **Attention heatmap (if TCN+Attention):** Which time steps receive the most attention weight, averaged across the test set

For each visualization, Claude must write 2-3 sentences of interpretation explaining what the chart reveals about the model's behavior.

### 6.5 Deployment-Readiness Checklist

Claude must produce this checklist and mark each item as PASS, FAIL, or N/A:

| # | Item | Status |
|---|------|--------|
| 1 | All data leakage tests pass | |
| 2 | Walk-forward validation used (not single split) | |
| 3 | Transaction costs included in backtest | |
| 4 | Model outperforms equal-weight on Sharpe by ≥ 0.15 | |
| 5 | Model outperforms at least 3 of 5 baselines on Sharpe | |
| 6 | Maximum drawdown ≤ 35% | |
| 7 | Model Sharpe > 0 in at least 2 of 3 test folds | |
| 8 | Seed sensitivity: Sharpe std < 0.3 across 5 seeds | |
| 9 | No degenerate weight behavior (concentration, collapse) | |
| 10 | Stress test results documented for all 3 crisis periods | |
| 11 | Feature ablation completed and documented | |
| 12 | All code reproducible with fixed seeds and config file | |
| 13 | Model comparison table complete with all metrics | |
| 14 | All visualizations generated and interpreted | |

---

## SECTION 7 — FINAL OUTCOME DEFINITION

### 7.1 What "Done" Means

The project is complete when ALL of the following are true:

1. **Reproducible results:** Any person can clone the repository, run the pipeline with the provided config, and obtain results within ±5% of the reported metrics (accounting for minor floating-point differences)

2. **Clean backtest:** The backtest uses next-day execution, includes transaction costs, respects all portfolio constraints, uses walk-forward validation with embargo periods, and contains zero instances of look-ahead bias

3. **Clear superiority:** The deep learning model outperforms the equal-weight baseline by at least 0.15 Sharpe ratio on the concatenated out-of-sample test period, and outperforms at least 3 of the 5 baseline strategies on Sharpe ratio

4. **Documented failures:** If the model does NOT achieve clear superiority, the project is still academically valid IF Claude has: (a) documented exactly which criteria failed and by how much, (b) provided hypotheses for why, (c) demonstrated that the methodology was rigorous regardless of outcome, (d) shown that the model provides some value even if not dominant (e.g., lower drawdown, better crisis behavior)

5. **Readiness for extension:** The codebase is modular enough that a future developer could: swap in different assets, add new features, change the model architecture, or extend to weekly/monthly rebalancing without rewriting the entire system

### 7.2 Project Directory Structure

Claude must organize the final codebase as follows:

```
project/
├── config.yaml                    # All hyperparameters, paths, seeds
├── README.md                      # Setup instructions, how to reproduce
├── requirements.txt               # Pinned dependency versions
├── data/
│   ├── raw/                       # Downloaded data (gitignored)
│   ├── processed/                 # Cleaned features (gitignored)
│   └── download.py                # Data acquisition script
├── features/
│   ├── engineering.py             # All feature computation
│   └── validation.py              # Leakage detection tests
├── models/
│   ├── tcn_attention.py           # TCN + Attention architecture
│   ├── lstm_allocator.py          # LSTM baseline
│   └── losses.py                  # Sharpe loss, augmented loss
├── training/
│   ├── trainer.py                 # Training loop, early stopping
│   ├── walk_forward.py            # Walk-forward validation pipeline
│   └── hyperparameter_search.py   # Optuna or manual search
├── baselines/
│   ├── equal_weight.py
│   ├── mean_variance.py
│   ├── momentum.py
│   ├── buy_and_hold.py
│   └── risk_parity.py
├── backtesting/
│   ├── engine.py                  # Backtest engine with costs
│   └── metrics.py                 # All metric computations
├── evaluation/
│   ├── comparison.py              # Model vs baseline comparison
│   ├── diagnostics.py             # All safeguard checks
│   ├── stress_test.py             # Crisis period analysis
│   └── visualizations.py          # All charts and plots
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_final_results.ipynb
└── results/
    ├── figures/                   # All generated plots
    ├── tables/                    # CSV metric tables
    └── checkpoints/               # Saved model weights
```

### 7.3 Key Technical Requirements

- **Python version:** 3.10+
- **Deep learning framework:** PyTorch (not TensorFlow — PyTorch is more flexible for custom loss functions and has better debugging)
- **Key libraries:** `yfinance`, `pandas`, `numpy`, `torch`, `optuna` (if used), `matplotlib`, `seaborn`, `scipy`, `scikit-learn` (for PCA only)
- **All random seeds must be set:** `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`
- **Deterministic mode:** `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`

---

## APPENDIX A — COMMON PITFALLS CLAUDE MUST AVOID

1. **Using future information for normalization.** Every z-score, min-max, or rank must use a strictly backward-looking window. Global statistics are NEVER acceptable.

2. **Training on the test set.** Even indirectly — if you look at test results and then "go back and tweak," you have contaminated the test set. The test set is used ONCE, at the very end.

3. **Reporting train-period metrics as results.** Only out-of-sample metrics count.

4. **Ignoring transaction costs.** A strategy with Sharpe 1.5 before costs and 200% monthly turnover is worthless. Costs must be in the training loss AND the backtest.

5. **Softmax temperature issues.** If the softmax output is nearly one-hot (all weight on one asset), the model has collapsed. Monitor the entropy of the weight vector; it should remain above 0.5 × ln(N).

6. **Survivorship bias.** Only including stocks that are currently in the DJIA, not ones that were removed. Document this limitation if historical composition is unavailable.

7. **Confusing Sharpe ratio calculation.** Use the correct annualization: multiply daily Sharpe by √252, not by 252. Excess return should ideally be computed over the risk-free rate, but for simplicity and comparability, using raw returns (assuming rf ≈ 0) is acceptable if documented.

8. **Cherry-picking the best seed.** Report the average across seeds, not the best run.

---

## APPENDIX B — GLOSSARY FOR CLAUDE

| Term | Definition |
|------|-----------|
| TAA | Tactical Asset Allocation — dynamically adjusting portfolio weights based on market conditions |
| Walk-forward | Training on expanding windows and testing on subsequent out-of-sample periods |
| Embargo | A gap between training and test data to prevent feature leakage from rolling calculations |
| Turnover | Sum of absolute changes in portfolio weights between rebalancing dates |
| Drawdown | Decline from a portfolio's peak value to its trough |
| Sharpe Ratio | Mean excess return divided by standard deviation of returns (annualized) |
| Calmar Ratio | Annualized return divided by maximum drawdown |
| Sortino Ratio | Like Sharpe but using only downside deviation |
| CVaR | Conditional Value at Risk — average loss in the worst X% of outcomes |
| TCN | Temporal Convolutional Network — uses dilated causal convolutions for sequence modeling |
| PPO | Proximal Policy Optimization — a stable policy gradient RL algorithm |
| Price Relative | close_t / close_{t-1} — the ratio of today's price to yesterday's |
| Softmax | Function that converts a vector of real numbers to a probability distribution |

---

*End of Execution Blueprint. Claude should begin at Phase 1, proceed sequentially, and never skip a gate.*
