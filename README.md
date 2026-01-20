![ Predictive-Electricity-Theft-Detection (%)](https://github.com/Sesery/Capstone-Proposal-Predictive-Electricity-Theft-Detection/blob/main/ElectricityTheftDetection.png)

# Capstone-Proposal-Predictive-Electricity-Theft-Detection

## 📋 Project Overview

This is an end-to-end machine learning system for detecting electricity theft using smart meter data. The project implements a supervised learning approach to identify fraudulent consumption patterns based on real-world theft behaviors. This is based on Institute of Electrical and Electronics Engineers (IEEE) research, the world's largest technical professional organization dedicated to advancing technology for the benefit of humanity.

## Business Objective

The primary business objective of this project is to develop a machine learning-driven system capable of detecting electricity theft with high precision and recall, directly enabling utility companies to reduce revenue losses stemming from non-technical losses. By accurately identifying fraudulent consumption patterns, the system allows for the efficient prioritization of field inspections, ensuring that investigative resources are allocated to the highest-risk cases. This facilitates proactive fraud detection, shifting from reactive investigations to preventative monitoring and timely intervention. Ultimately, the implementation of this solution enhances grid security and operational efficiency, safeguarding revenue, ensuring equitable billing, and contributing to a more stable and reliable electricity distribution infrastructure.

## 📊 Dataset understanding

* Source: UCI ElectricityLoadDiagrams2011-2014.
* Description: 370 smart meters with 15-minute interval readings from 2011-2014.
* Original Size: 370 columns × 140,256 time points (~4 years of data).
* Processed: Aggregated to daily consumption per meter (540,940 records)in kilowatt-hours.

![Global distribution of consumption (%)](https://github.com/Sesery/Capstone-Proposal-Predictive-Electricity-Theft-Detection/blob/main/Images/global%20consumption.png)

## 🏗️ System Architecture

The system architecture is designed as a robust, end-to-end machine learning pipeline specifically tailored for electricity theft detection. It begins with a comprehensive data processing pipeline that transforms raw 15-minute interval smart meter data into insightful daily consumption profiles. This stage involves sophisticated feature engineering, generating over 30 distinct features that capture critical behavioral patterns. These features quantify statistical properties, rolling temporal statistics across 7, 30, and 90-day windows, pattern consistency through autocorrelation and seasonality, anomaly indicators like sudden drops, and fraud-specific signals such as Benford's Law violations.

![Global distribution of consumption (%)](https://github.com/Sesery/Capstone-Proposal-Predictive-Electricity-Theft-Detection/blob/main/Images/Anomalies.png)

Correlation analysis reveals two clearly distinct signal groups:

1. Level-based metrics (7-day and 30-day z-scores) are highly correlated with each other (~0.79) but weakly correlated with theft-related instability metrics.

2. Instability metrics—especially volatility, maximum daily drops, and sudden change counts—form a separate behavioral signature.
   
To enable supervised learning on a dataset lacking real fraud labels, a realistic theft injection mechanism was implemented, grounded in IEEE research. This mechanism synthetically creates five distinct fraud types with controlled prevalence: Meter Tampering (35%), involving systematic under-recording; Cable Bypass (25%), representing complete meter circumvention; Partial Bypass (20%), for partial load shunting; Time-Based Theft (15%), occurring during specific low-inspection periods; and Gradual Theft (5%), featuring slowly increasing manipulation. These patterns are injected into 5% of the customer base, creating a labeled, imbalanced dataset that mirrors real-world theft distributions.

The model architecture employs a strategic ensemble approach to maximize detection performance. The model stack incorporates a Logistic Regression baseline for interpretability and linear pattern capture, a Random Forest classifier to leverage feature importance and non-linear interactions, and an optimized XGBoost model fine-tuned with early stopping for superior gradient-boosting performance. These diverse algorithms are combined into a final Voting Classifier utilizing weighted soft voting, which synthesizes their individual strengths to produce a more robust and accurate prediction than any single model could achieve alone.

Finally, a multi-faceted evaluation framework ensures the solution meets both technical and business requirements. The primary metric is the F2-Score, which deliberately emphasizes recall to minimize missed theft cases—a critical business imperative. Complementary business metrics include Precision@10%, which measures how effectively the model prioritizes the highest-risk customers for field inspections. This is supplemented by comprehensive standard metrics like accuracy, precision, recall, and PR-AUC, providing a holistic view of model performance across all operational dimensions relevant to utility company deployment.

![Global distribution of consumption (%)](https://github.com/Sesery/Capstone-Proposal-Predictive-Electricity-Theft-Detection/blob/main/Images/Architecture.png)

## 📈 Key Results
The Ensemble model has been selected as the final solution due to its superior performance in identifying energy theft. It achieves the best balance across key metrics, especially by optimizing the F2-Score to prioritize catching as many theft cases as possible (high recall) while maintaining solid precision. With leading PR AUC and a strong Precision@10%, it ensures inspectors can efficiently target the top high-risk accounts, where nearly half of true thefts are found. By combining different model types and carefully tuning its parameters, it effectively handles the rare-event challenge, making it the most reliable and operationally ready choice for deployment.

![End-to-end data processing workflow (%)](https://github.com/Sesery/Capstone-Proposal-Predictive-Electricity-Theft-Detection/blob/main/Images/Model%20comparison.jpeg)

## Top Features for Detection
The detection system utilizes five key features to identify anomalous energy consumption patterns. First, the z_score_90d captures long-term deviations by measuring how far recent consumption strays from a 90-day normal pattern. Second, cum_dev_last focuses on more recent discrepancies by tracking the short-term departure from the historical mean. To uncover data integrity issues, the benford_violation feature analyzes digit distribution for statistical irregularities that may suggest manipulation. Fourth, autocorr_weekly detects disruptions in expected weekly cyclical patterns, indicating a break in habitual usage. Finally, the sudden_drop_count quantifies the frequency of abrupt consumption decreases, signaling potential faults or tampering events. Together, these features provide a multi-faceted analytical approach for robust anomaly detection.

Based on the confusion matrix, the model demonstrates exceptionally strong overall accuracy (98.6%) by correctly identifying nearly all normal transactions (69 True Negatives). It also maintains a very low false alarm rate, with only one normal transaction being incorrectly flagged as theft (1 False Positive). However, while it successfully detects 75% of theft cases (3 True Positives), it misses one out of four actual thefts (1 False Negative), indicating that while the system is highly reliable for routine monitoring, there remains a moderate risk of undetected fraud that requires attention.

![End-to-end data processing workflow (%)](https://github.com/Sesery/Capstone-Proposal-Predictive-Electricity-Theft-Detection/blob/main/Images/confusion%20matrix.png)

## 🛠️ Technical Implementation

### Feature Engineering Highlights
These four methods provide a robust framework for analyzing energy consumption data to detect anomalies;

* Benford's Law Analysis: Benford's Law Analysis identifies unnatural patterns in the distribution of meter reading digits, which can indicate data manipulation.

* Rolling Statistics: monitor each customer's usage over time, flagging deviations from their personal historical norms.

* Pattern Consistency: using autocorrelation, checks for disruptions in regular weekly or daily consumption habits that characterize habitual behavior.

* Entropy Measures: quantify the randomness or predictability of usage; unusually high or low entropy can signal irregular consumption.
Together, these techniques create a multi-layered detection system sensitive to both subtle inconsistencies and overt fraudulent activity.

### Model Optimization

To ensure the model is both effective and reliable, we implemented several key technical steps to address common challenges in machine learning. Class Imbalance was tackled through a dual approach, combining SMOTE oversampling to artificially create more theft examples with class weighting to make the model pay more attention to the minority class during training. 

Hyperparameter Tuning via a systematic grid search allowed us to find the optimal model settings for maximum performance. Within the XGBoost algorithm, Early Stopping was used to halt training once performance on a validation set stopped improving, which effectively prevents the model from overfitting to the training data. 

Finally, all numerical features were processed with Feature Scaling using StandardScaler, which standardizes the data range to ensure numerical stability and help algorithms like Logistic Regression converge more efficiently.


## 📊 Business Impact

![End-to-end data processing workflow (%)](https://github.com/Sesery/Capstone-Proposal-Predictive-Electricity-Theft-Detection/blob/main/Images/Financial%20impact.png)

Our business impact analysis reveals the model's exceptional financial and operational efficiency. At an optimized decision threshold of approximately 0.17, the Ensemble model targets only the top 3.75% of customers (375,000 inspected) to capture 75% of all theft cases. This precise targeting yields a gross annual recovery of KES 13.5 billion at an inspection cost of KES 187.5 million, resulting in a net gain of KES 13.31 billion and a remarkable 71x return on investment (ROI). The strategy's precision is further evidenced by a Precision@10% of 43%, meaning that among the 10% of customers flagged as most suspicious, nearly half are confirmed thefts, ensuring that inspection resources are deployed with high confidence and maximum efficiency.

## 📊 Distribution of theft categories

![End-to-end data processing workflow (%)](https://github.com/Sesery/Capstone-Proposal-Predictive-Electricity-Theft-Detection/blob/main/Images/Theft%20categories.png)

The bar chart shows a highly imbalanced distribution of theft risk across meters.
The vast majority of meters (95.1%) are classified as Low Risk, indicating that most customers exhibit normal consumption patterns.

Only a small fraction fall into Medium Risk (2.7%) and High Risk (2.2%) categories.
Although these groups are small in proportion, they are operationally significant, as they represent the highest-priority candidates for targeted inspection and fraud investigation.

This visualization highlights the importance of precision-oriented models and selective intervention strategies, rather than broad, resource-intensive enforcement.

## 🔮 RECOMMENDATION & CONCLUSION

The proposed solution centers on deploying an Ensemble model, which effectively captures the diverse patterns of electricity theft. Operating at an optimized threshold, it targets a highly focused 3.75% of customers, enabling the recovery of KES 13.5 billion annually at minimal cost, achieving an exceptional 71x ROI. To sustain this impact, strategic actions include enriching the dataset with external features, segmenting customers by risk, and implementing a dynamic KPI dashboard for continuous monitoring and evaluation. By prioritizing high-risk inspections and annually adjusting the model, the strategy ensures long-term operational efficiency, minimizes residual losses, and maximizes financial recovery within practical field capacity.

## 👥 Team & Contact
Kelvin Sesery

Sharon Thiga

Victor Wasunna

Ann Wahu

Elizabeth Gichure

Joan Omanyo

GitHub: [Repository Link]














