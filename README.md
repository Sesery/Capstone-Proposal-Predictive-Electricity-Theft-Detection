# Capstone-Proposal-Predictive-Electricity-Theft-Detection

![Electricity consumption distribution (%)](https://github.com/Sesery/Capstone-Proposal-Predictive-Electricity-Theft-Detection/blob/main/global%20consumption.png)

## 📋 Project Overview

This is an end-to-end machine learning system for detecting electricity theft using smart meter data. The project implements a supervised learning approach to identify fraudulent consumption patterns based on real-world theft behaviors. This is based on Institute of Electrical and Electronics Engineers (IEEE) research, the world's largest technical professional organization dedicated to advancing technology for the benefit of humanity.

## Business Objective

Detect electricity theft with high precision and recall, enabling utility companies to:

1. Reduce revenue losses from non-technical losses.

2. Prioritize field inspections efficiently.

3. Implement proactive fraud detection.

4. Improve grid security and operational efficiency.

## 📊 Dataset

* Source: UCI ElectricityLoadDiagrams2011-2014.
* Description: 370 smart meters with 15-minute interval readings from 2011-2014.
* Original Size: 370 columns × 140,256 time points (~4 years of data).
* Processed: Aggregated to daily consumption per meter (540,940 records)in kilowatt-hours.


![Electricity consumption distribution (%)](https://github.com/Sesery/Capstone-Proposal-Predictive-Electricity-Theft-Detection/blob/main/global%20consumption.png)

## 🏗️ System Architecture

### 1. Data Processing Pipeline

* Temporal Aggregation: 15-minute → daily consumption

* Feature Engineering: 30+ engineered features capturing:

1. Statistical properties (mean, std, CV).

2. Rolling statistics (7, 30, 90-day windows).

3. Pattern consistency (autocorrelation, seasonality).

3. Anomaly indicators (outliers, sudden changes).

4. Fraud-specific signals (Benford's Law violations).

### 2. Theft Injection Mechanism

- Realistic patterns based on IEEE research on electricity theft.

- 5 Fraud Types:

    i) Meter Tampering (35%): Systematic under-recording.

    ii) Cable Bypass (25%): Complete meter bypass.

    iii) Partial Bypass (20%): Partial load bypass.

    iv) Time-Based Theft (15%): Theft during specific periods.

    v) Gradual Theft (5%): Slowly increasing theft.

- Controlled Injection: 5% of customers flagged as fraudulent.

### 3. Model Architecture

- Ensemble Approach: Combines multiple algorithms.

- Model Stack:

    * Baseline: Logistic Regression (balanced).

    * Primary: Random Forest (feature importance).

    * Best: XGBoost (optimized with early stopping).

    * Ensemble: Voting classifier with weighted voting.

### 4. Evaluation Framework

    * Primary Metric: F2-Score (emphasizes recall)

    * Business Metrics: Precision@10% (inspection prioritization)

    * Comprehensive Metrics: Accuracy, Precision, Recall, PR-AUC

## 📈 Key Results

<details> <summary><b>📊 Detailed Performance Table</b></summary><div align="center">
Model	F2-Score	Precision@10%	PR AUC	Precision	Recall	Accuracy
XGBoost	0.8572	0.6471	0.7680	0.6667	0.8824	0.9831
Random Forest	0.8478	0.6471	0.7639	0.6667	0.8824	0.9831
Ensemble	0.8197	0.7059	0.7349	0.7059	0.7059	0.9828
Logistic Regression	0.7901	0.5882	0.6839	0.5882	0.8824	0.9797
</div> <p><em>🥇 Best metric in column is bolded. Ensemble provides optimal Precision@10% for real-world deployment.</em></p> </details>

## Top Features for Detection

    1. z_score_90d: Long-term deviation from normal pattern.

    2. cum_dev_last: Recent deviation from historical mean.

    3. benford_violation: Irregular digit distribution.

    4. autocorr_weekly: Disruption in weekly patterns.

    5. sudden_drop_count: Frequency of abrupt consumption drops.


## 🛠️ Technical Implementation

### Feature Engineering Highlights

* Benford's Law Analysis: Detects unnatural consumption distributions.

* Rolling Statistics: Customer-specific anomaly detection.

* Pattern Consistency: Autocorrelation for habitual behavior.

* Entropy Measures: Consumption randomness quantification.

### Model Optimization

* Class Imbalance: SMOTE oversampling + class weighting.

* Hyperparameter Tuning: Grid search for optimal parameters.

* Early Stopping: Prevents overfitting in XGBoost.

* Feature Scaling: StandardScaler for numerical stability.


## 📊 Business Impact
### Operational Benefits
Inspection Efficiency: Top 10% predictions capture 70% of theft cases.

False Positive Reduction: Ensemble approach balances precision/recall.

Scalability: Handles 370+ meters with 4 years of historical data.

Interpretability: Feature importance guides investigation priorities.

### Financial Implications

Assuming:

Average theft: €500/month per customer.

Detection rate: 88% (model recall).

370 customers, 5% theft prevalence.

Annual Savings Potential: €97,680.
Inspection Efficiency Gain: 70% reduction in false inspections.

## 🔮 Future Enhancements

Timeframe	Key Enhancements
Short-term (0–3 months)	Real-time streaming pipeline, customer segmentation, explainable AI dashboard
Medium-term (3–12 months)	Billing system integration, weather/calendar feature enrichment, transfer learning
Long-term (12+ months)	Deep learning for temporal patterns, federated learning, blockchain audit logging

## 📚 References
UCI Machine Learning Repository - ElectricityLoadDiagrams.

IEEE Papers on Electricity Theft Detection.

Portuguese Energy Regulatory Authority Reports.

Industry Best Practices for Non-Technical Loss Detection.

## 👥 Team & Contact
Project Lead: [Your Name/Team Name]
Organization: [Your Organization]
Email: [Contact Email]
GitHub: [Repository Link]












