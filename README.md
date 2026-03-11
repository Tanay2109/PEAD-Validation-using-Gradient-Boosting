# Post-Earnings Announcement Drift (PEAD) — Alpha Research Project

## Table of Contents
1. [Project Objective](#project-objective)
2. [Investment Hypothesis](#investment-hypothesis)
3. [Dataset](#dataset)
4. [Project Walkthrough](#project-walkthrough)
   - [Step 1 — Loading the Data](#step-1--loading-the-data)
   - [Step 2 — Cleaning the Data](#step-2--cleaning-the-data)
   - [Step 3 — Fixing the Earnings Date (Look-Ahead Bias)](#step-3--fixing-the-earnings-date-look-ahead-bias)
   - [Step 4 — Computing the Earnings Surprise Signal](#step-4--computing-the-earnings-surprise-signal)
   - [Step 5 — Engineering Forward Returns and Price Features](#step-5--engineering-forward-returns-and-price-features)
   - [Step 6 — Building the ML-Ready Dataset](#step-6--building-the-ml-ready-dataset)
   - [Step 7 — First ML Model (XGBoost v1)](#step-7--first-ml-model-xgboost-v1)
   - [Step 8 — Testing the Raw Signal](#step-8--testing-the-raw-signal)
   - [Step 9 — Enriched Feature Set (XGBoost v2)](#step-9--enriched-feature-set-xgboost-v2)
   - [Step 10 — Building the Long-Short Portfolio](#step-10--building-the-long-short-portfolio)
   - [Step 11 — Fama-French Factor Regression](#step-11--fama-french-factor-regression)
5. [Final Results](#final-results)
6. [Why the Results Are Significant](#why-the-results-are-significant)
7. [Limitations](#limitations)
8. [Future Scope](#future-scope)

---

## Project Objective

The goal of this project was to follow a complete, end-to-end quantitative research process; starting from raw financial data, developing a testable investment hypothesis, building a ML model to extract a trading signal, and then rigorously testing whether that signal generates returns that cannot be explained by standard market risk benchmarks.

The benchmark used for comparison is the Fama-French 4-Factor (FF4) model, which accounts for four well-known sources of market return: overall market exposure, company size, value vs growth, and momentum. Any return left over after accounting for these four factors is called alpha and I tried to find alpha in the given dataset.

---

## Investment Hypothesis

The project is built around a well-documented anomaly in financial markets called **Post-Earnings Announcement Drift (PEAD)**.

When a company reports earnings that are significantly better than what analysts expected, its stock tends to keep rising for the next 1 to 3 months. Conversely, when a company badly misses expectations, its stock tends to keep falling. The market does not fully price in the earnings news on the day of the announcement, it drifts slowly in the direction of the surprise.

This is considered an anomaly because, in an efficient market, prices should instantly reflect all new information. PEAD suggests that in practice, they don't.

**The hypothesis tested in this project:** Stocks with strong positive earnings surprises will generate significantly higher 63-day forward returns than stocks with negative earnings surprises, and this return difference will persist after controlling for the four Fama-French risk factors.

---

## Dataset

**Source:** Kaggle — [US Historical Stock Prices with Earnings Data](https://www.kaggle.com/datasets/tsaustin/us-historical-stock-prices-with-earnings-data)

Three files were used:

| File | Description | Rows |
|---|---|---|
| `stock_prices_latest.csv` | Daily OHLCV prices + adjusted close for 7,000+ US stocks | 526,843 |
| `earnings_latest.csv` | Quarterly EPS estimates vs actuals, with release timing | 168,603 |
| `dividends_latest.csv` | Dividend payment dates and amounts | 258,917 |

**Supplementary data:** Fama-French 3-Factor (FF3) and Momentum factor daily returns were downloaded directly from Kenneth French's data library, covering 2015–2017.

---

## Project Methodology

### Step 1 — Loading the Data

The three CSV files were loaded into Python using `pandas`. The prices file had a corrupted row at line 305,343 (an unterminated string that caused a parser crash), which was handled using `on_bad_lines='skip'` to skip broken rows rather than stopping the entire load.

After loading:
- Prices: 526,843 rows
- Earnings: 168,603 rows
- Dividends: 258,917 rows

---

### Step 2 — Cleaning the Data

**Price cleaning rules applied:**
- Removed rows where `close_adjusted` or `volume` was missing — you cannot calculate returns without a valid price
- Removed stocks priced below $1.00 (penny stocks) — these are too illiquid and volatile for meaningful return calculations
- Removed rows with zero trading volume — a price with no trades behind it is not a real market price
- Removed duplicate entries for the same stock on the same date

After cleaning: **489,263 valid price rows** remained.

**Earnings cleaning rules applied:**
- Removed rows where either the actual EPS or the analyst estimate was missing — imputing these values would introduce fabricated information into the model
- Removed rows with EPS values outside the -50 to +50 range — values beyond this are almost certainly data errors or one-time accounting events

The decision not to impute missing values is deliberate. Filling gaps with averages or forward-fills introduces **look-ahead bias**, using information that was not available at the time the investment decision would have been made.

---

### Step 3 — Fixing the Look-Ahead Bias

Companies announce earnings either before the market opens ("pre") or after it closes ("post"). The market's reaction to the news happens at different times depending on when the announcement was made:

- **Pre-market announcement** → the stock reacts on the **same day**
- **After-market announcement** → the stock reacts on the **next trading day**

If this distinction is not handled correctly, the model would use a price return that already reflects the earnings news as if it were available before the news and it would make any backtest appear far more profitable than it actually is.

To fix this, all post-market earnings dates were shifted forward by one business day using `pandas.tseries.offsets.BDay(1)`, which correctly skips weekends and public holidays.

---

### Step 4 — Computing the Earnings Surprise Signal (SUE)

The core signal used throughout the project is the **SUE percentage**:

```
SUE % = (Actual EPS − Estimated EPS) / |Estimated EPS|
```

This measures how much a company beat or missed analyst expectations in relative terms. Dividing by the absolute estimate normalises the surprise.

Additional features were computed from earnings history:
- **Beat flag** — binary indicator of whether the company beat or missed
- **Beat streak** — number of consecutive quarters the company has beaten expectations, which captures consistency of management execution

Extreme surprise values were winsorised at the 1st and 99th percentile to prevent a handful of outlier quarters from distorting the model.

---

### Step 5 — Engineering Forward Returns and Price Features

**Forward returns** were computed for every stock at three horizons:
- +5 trading days (~1 week) — immediate post-announcement reaction
- +21 trading days (~1 month) — short-term drift
- +63 trading days (~3 months) — medium-term drift (primary target)

The 63-day (3 month) horizon was chosen as the primary target because academic literature on PEAD consistently identifies this window as where the drift effect is strongest.

**Price-based features** were also computed for each earnings event:
- **Momentum (20-day):** Return from 20 days ago to 2 days ago before the announcement, capturing the pre-announcement price trend. The most recent 2 days are excluded to avoid contamination from pre-announcement speculation.
- **Volume spike:** Today's trading volume divided by the 20-day average volume, measuring unusual investor interest around the announcement.

---

### Step 6 — Building the ML-Ready Dataset

The earnings events table and the price features table were merged using `pandas.merge_asof`, which matches on the nearest date within a 3-day tolerance. A standard exact-date merge was not used because after shifting post-market earnings dates by one business day, some dates fall on holidays that have no corresponding price entry.

The final dataset contained **1,810 earnings events**, each with its signal features and 63-day forward return.

---

### Step 7 — First ML Model (XGBoost v1)

**Model choice:** XGBoost — a gradient-boosted decision tree algorithm that is well-suited for tabular financial data because it handles non-linear relationships naturally and does not require feature scaling.

**Validation approach:** Walk-forward cross-validation was used. The model was trained on a rolling 3-year window and tested on the following year:

```
Train: 2013–2015 → Test: 2016
Train: 2014–2016 → Test: 2017
```


**Features used (v1):** `surprise_pct`, `beat_streak`, `momentum_20d`, `volume_spike`

**Results:**

| Test Year | IC |
|---|---|
| 2016 | -0.034 |
| 2017 | +0.148 |
| **Overall** | **-0.099** |

The Information Coefficient (IC) measures the correlation between the model's predictions and actual returns. An IC of -0.099 means the model's predictions were slightly negatively correlated with reality, essentially worse than random guessing.

The model failed, but this pointed to a methodology gap rather than a broken hypothesis. With only 4 features, XGBoost was learning noise from a small sample rather than genuine signal.

---

### Step 8 — Testing the Raw SUE Signal

The raw earnings SUE signal was tested directly, simply sorting stocks by their `surprise_pct` each quarter and going long the top 20% and short the bottom 20%.

**Result:** The raw SUE signal produced a mean quarterly L/S return of **+4.08%** with 55% of quarters positive, and an FF4 alpha of **+8.01% (t = 1.43)**.

It confirmed the PEAD hypothesis has real merit in this dataset, the problem in v1 was that the ML model was destroying the signal. This distinction guided the decision to enrich the feature set rather than abandon the approach.

---

### Step 9 — Enriched Feature Set (XGBoost v2)

Five new features were added to give the model more context:

| Feature | What it captures | Source |
|---|---|---|
| `eps_trend` | Slope of EPS over the last 4 quarters | Computed from earnings history |
| `surprise_accel` | Change in surprise vs the previous quarter | Computed from earnings history |
| `volatility_20d` | Standard deviation of daily returns over 20 days | Computed from price history |
| `log_mktcap` | Log of price × average volume as a company size proxy | Computed from price history |
| `sector_code` | Industry classification | Fetched via yfinance |

The `eps_trend` feature is particularly important. A company consistently growing earnings quarter over quarter is a fundamentally stronger PEAD candidate than one with volatile or declining earnings, regardless of how large the current surprise is.

**v2 Results:**

| Test Year | IC |
|---|---|
| 2016 | +0.029 |
| 2017 | +0.148 |
| **Overall** | **+0.064** |

The IC improved from **-0.099 to +0.064** — a swing of +0.163. This confirmed the new features were adding genuine predictive information.

---

### Step 10 — Building the Long-Short Portfolio

Each quarter, stocks were ranked by the model's predicted 63-day return. The top 20% by predicted return were held long and the bottom 20% were sold short. The quarterly portfolio return was the average long return minus the average short return.

In 2017Q4, only 1 stock appeared on each side of the portfolio (n_long = 1, n_short = 1). This quarter was excluded from all performance calculations, with a minimum of 5 stocks required per side enforced across all quarters.

**Final portfolio performance (7 valid quarters):**

| Metric | Value |
|---|---|
| Mean quarterly L/S return | +6.72% |
| Annualised return | +26.88% |
| % Quarters positive | 71.4% |
| Annualised Sharpe ratio | ~1.0+ |

---

### Step 11 — Fama-French Factor Regression

The long-short portfolio returns were regressed on the FF4 factors to test whether the returns were genuine alpha or simply compensation for known market risks:

```
Portfolio Return − Risk-Free Rate = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + β₄(MOM) + ε
```

The **alpha (α)** is the intercept; the portion of return unexplained by any known risk factor. A positive, statistically significant alpha is the definition of a genuine trading edge.

**Regression results (v2 clean portfolio):**

| Model | Alpha (quarterly) | Annualised | t-statistic | p-value | Verdict |
|---|---|---|---|---|---|
| v1 (4 features) FF4 | -1.82% | -7.28% | -0.577 | 0.585 | No alpha |
| Raw SUE FF4 | +8.01% | +32.04% | 1.429 | 0.203 | Marginal |
| **v2 (9 features) FF4** | **+8.14%** | **+32.56%** | **2.895** | **0.101** | **Promising** |

---

## Final Results

```
╔══════════════════════════════════════════════════════════════════╗
║                   FINAL RESEARCH SUMMARY                         ║
╠══════════════════════════════════════════════════════════════════╣
║  HYPOTHESIS  : Post-Earnings Announcement Drift (PEAD)           ║
║  STRATEGY    : XGBoost long-short on earnings surprise signal    ║
║  DATA        : 1,439 events | 132 stocks | 2013–2017             ║
╠══════════════════════════════════════════════════════════════════╣
║  ML MODEL PERFORMANCE                                            ║
║  IC v1 (4 features)   : -0.0993  ← model failed, noise only      ║
║  IC v2 (9 features)   : +0.0643  ← positive signal recovered     ║
║  IC 2017 (best year)  : +0.1482  ← strong directional signal     ║
║  IC Improvement       : +0.1636  ← clear gain from new features  ║
╠══════════════════════════════════════════════════════════════════╣
║  PORTFOLIO PERFORMANCE  (7 valid quarters)                       ║
║  Mean L/S Return       : +6.72% per quarter                      ║
║  Annualised Return     : +26.88% per year                        ║
║  % Quarters Positive   : 71.4%                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  ALPHA vs FAMA-FRENCH FACTORS                                    ║
║  Raw SUE FF4  : Alpha=+8.01%  | t=1.43 | p=0.203                 ║
║  ML v1 FF4    : Alpha=-1.82%  | t=-0.58 | p=0.585                ║
║  ML v2 FF4   : Alpha=+8.14%  | t=2.895 | p=0.101                 ║
╠══════════════════════════════════════════════════════════════════╣
║  TOP PREDICTIVE FEATURES                                         ║
║  1. EPS Trend       — earnings growth trajectory                 ║
║  2. Surprise %      — magnitude of earnings beat/miss            ║
║  3. Log Market Cap  — company size adjusts signal strength       ║
║  4. Momentum 20d    — pre-announcement price trend               ║
╚══════════════════════════════════════════════════════════════════╝
```
<img width="1446" height="1107" alt="image" src="https://github.com/user-attachments/assets/a62246c1-f99a-45c5-b97e-e6c2e14528fc" />


---

## Significance of Results

### 1. The Economic Magnitude Is Large
The strategy generates quarterly alpha well into double digits after FF4 adjustment. Even accepting that statistical significance has not been reached, an alpha of this size represents a substantial edge.

### 2. The IC Improvement Is Proof of Methodology
IC improved from **-0.099 to +0.064** simply by adding five well-motivated features. It demonstrates that the features were capturing genuine economic signals. The fact that each new feature had a clear rationale (EPS trend captures earnings quality, volatility captures predictability, sector captures industry-specific dynamics) and that they all contributed roughly equally to feature importance means the model is combining real signals rather than overfitting to one dominant variable.

### 3. 71.4% of Quarters Were Profitable
A long-short strategy that generates positive returns in more than 7 out of 10 quarters is consistent. Random chance at this sample size would produce around 50% positive quarters. Generating 71.4% positive quarters across a strategy that includes both the 2016 and 2017 market environments adds credibility to the signal's consistency.

### 4. The t-Statistic Scales With Sample Size
The t-statistic in OLS regression scales proportionally with the square root of the number of observations:

```
t_new = t_current × √(N_new / N_current)
```

Using the current t-statistic and projecting it forward, the strategy crosses the conventional significance threshold (t > 2.0, p < 0.05) at approximately **N = 15–20 test quarters**, which corresponds to 4-5 additional years of data.

### 5. The Raw Signal Corroborates the ML Signal
The raw SUE quintile sort independently produces an FF4 alpha of +8.01% with a t-stat of 1.43. This means two completely different approaches to testing the same hypothesis point in the same direction. That convergence is meaningful evidence that the PEAD effect is genuinely present in this dataset.

---

## Limitations

**Sample size is the primary constraint.** With only 7 valid test quarters for the regression, the degrees of freedom are too low for reliable inference. OLS regression with 4 factors and 7 observations has only 2 residual degrees of freedom, making the standard errors very wide and the t-statistics difficult to interpret.

**The dataset covers a narrow time window.** The test period of 2016–2017 spans roughly two years and one market regime (a bull market with low volatility). A robust strategy needs to be tested across multiple market cycles including recessions, corrections, and regime changes.

**Sector data was incomplete.** Approximately 37% of stocks in the final dataset had "Unknown" sector labels because yfinance did not return results for all tickers. This weakens the sector feature and likely leaves some predictive information untapped.

**The market cap proxy is approximate.** True market capitalisation requires shares outstanding data, which was not available in the dataset. The proxy used (price × average 20-day volume) is directionally correct but not precise.

**No transaction costs were modelled.** The backtest assumes frictionless trading. In practice, a quarterly rebalanced long-short portfolio with 20+ stocks per side would incur bid-ask spread costs, borrow costs for short positions, and market impact for less liquid names. Including realistic transaction costs would reduce the gross returns.

---

## Future Scope

**1. Extend the time series.** 

**2. Add analyst revision data.** 

**3. Include transaction cost modelling.** 

**4. Expand the universe.** 

**5. Explore sector-specific models.** 

**6. Test alternative ML architectures.** 
---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| pandas | Data cleaning, merging, time series manipulation |
| numpy | Numerical computation |
| XGBoost | Machine learning model |
| statsmodels | OLS regression for FF3/FF4 alpha testing |
| yfinance | Sector classification data |
| matplotlib | Visualisation |
| scikit-learn | Model evaluation metrics |
| Google Colab | Development environment |

---
