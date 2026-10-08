# demand-forecasting-uncertainty
Quantile forecasting with calibrated prediction intervals on M5 Walmart data

# 📈 M5 Uncertainty Demand Forecasting & Cost-Sensitive Optimization

An end-to-end supply chain demand forecasting pipeline built on the **M5 Competition dataset**. This project uses **LightGBM Quantile Regression**, **Post-Hoc Conformal Calibration**, and an **Asymmetric Business Cost Matrix** to convert probabilistic predictions into actionable inventory optimization decisions.

---

## 📌 Executive Summary & Key Results

- **Cost Reduction:** Achieved a **~21.8% reduction in total operational inventory costs** using a cost-sensitive $P_{90}$ upper-bound quantile forecast compared to a standard point forecast ($P_{50}$).
- **Interval Coverage:** Post-hoc conformal calibration improved test set $P_{90}$ coverage to **90.0%**, perfectly aligning with the target 90% confidence level.
- **Asymmetric Risk Management:** Penalized understocking ($c_{\text{under}} = \$5$) five times higher than overstocking ($c_{\text{over}} = \$1$) to prevent stockouts during high-demand regimes.

---

## 🏗️ Architecture & Methodology




[Raw M5 Data] ➡️ [Feature Engineering (Lags 28+, Promos, Prices)] ➡️ [LightGBM Quantile Regressor (P10, P50, P90)]
⬇️
[Decision Matrix / Fill Rate Optimization] ⬅️ [Conformal Calibration (Coverage Tuning)]



1. **Non-Leakage Feature Engineering:** Engineered rolling lag features shifted by $\ge 28$ days to mirror real-world 28-day advance planning horizons without temporal leakage.
2. **Quantile Regression:** Trained LightGBM models using asymmetric pinball loss to predict $P_{10}$, $P_{50}$ (median demand), and $P_{90}$ prediction bounds.
3. **Conformal Calibration:** Applied post-hoc calibration on validation residuals to adjust upper prediction interval boundaries for exact coverage targets.
4. **Cost-Sensitive Inventory Decision Matrix:** Evaluated predictions against financial loss metrics:

$$\text{Total Cost} = (c_{\text{under}} \times \text{Units Short}) + (c_{\text{over}} \times \text{Units Leftover})$$

---

## 📊 Experimental Results

### Point Forecast Accuracy (Test Set)
| Strategy | MAE | RMSE |
| :--- | :---: | :---: |
| Baseline (Same as 28 days ago) | 2.35 | 5.14 |
| LightGBM $P_{50}$ | **1.89** | **4.08** |

### Inventory Decision Optimization
| Strategy | Total Cost ($) | Fill Rate (%) |
| :--- | :---: | :---: |
| Point Forecast ($P_{50}$) | $54,887 | 69.0% |
| $P_{50}$ + 20% Safety Buffer | $47,282 | 76.4% |
| Quantile $P_{90}$ | $42,900 | 94.2% |
| **Calibrated $P_{90}$** | **$43,764** | **94.6%** |

### Quantile Regression vs. Residual-Based Intervals
| Method | Coverage (80% target) | Width | $P_{90}$ Below-Share (90% target) | Pinball $P_{90}$ |
| :--- | :---: | :---: | :---: | :---: |
| Residual-Based Interval | 0.785 | 4.23 | 0.899 | 0.691 |
| Quantile Regression | 0.876 | 6.86 | 0.887 | 0.597 |
| **Quantile + Calibrated** | **0.890** | **7.05** | **0.900** | **0.595** |

---

## 🔬 Diagnostic Analysis & Known Failure Modes

Through stratified error analysis, the pipeline revealed **conditional overconfidence during peak demand spikes**:
- **Normal Days Coverage:** ~94.1% (reliable prediction bounds)
- **Spike Days Coverage:** ~29.9% (interval width narrows during sudden, unobserved demand surges)

**Root Causes Identified:**
- **Information Lag:** Strict 28-day lag constraints blind the model to short-term momentum preceding a spike.
- **Quantile Loss Smoothing:** Tree-based pinball loss optimization prioritizes fitting baseline demand days (~90%+ of history), smoothing over extreme tail risks.
- **Item Velocity Impact:** Fast-moving items exhibited lower coverage (76.5% under residual method vs. 87.9% under Quantile Regression) compared to slow-moving, intermittent items.

---

## 🚀 Future Roadmap (Phase 2 Improvements)

1. **Stratified / Conditional Conformal Calibration:** Apply separate calibration residuals ($q_{\text{hat}}$) partitioned by predicted volatility (Normal vs. Event/Spike Days).
2. **Event Proximity Features:** Engineer explicit countdown/count-up features (`days_to_event`, `days_after_event`) to help decision trees isolate high-variance regimes.
3. **Two-Stage Hurdle Model:** Deploy a Stage-1 spike classifier paired with conditional quantile regression.

---

## 💻 Installation & Usage


# Clone repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME

# Install requirements
pip install -r requirements.txt

# Run main pipeline notebook
jupyter notebook notebook.ipynb
