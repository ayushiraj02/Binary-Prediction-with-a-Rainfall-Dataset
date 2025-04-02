# 🏆 Kaggle Playground Series: Backpack Price Prediction

This repository contains my solution for the **Kaggle Playground Series - Binary Prediction with a Rainfall Dataset** competition.

### Link of Competition: https://www.kaggle.com/competitions/playground-series-s5e3


[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/playground-series-s5e3)
![Rank](https://img.shields.io/badge/Rank-520%2F4382-blue)
![Score](https://img.shields.io/badge/Score-0.89823_AUC-brightgreen)

## Competition Overview
**Task:** Binary classification to predict rainfall occurrence  
**Evaluation Metric:** ROC AUC (Area Under the ROC Curve)  
**Final Rank:** 520/4382 participants  
**Best Score:** 0.89823 AUC (Public LB)  

![Image description](https://raw.githubusercontent.com/ayushiraj02/Binary-Prediction-with-a-Rainfall-Dataset/refs/heads/main/img.png)

---

## Approach
### Key Technical Strategies
- **Temporal Feature Engineering**
  - Cyclical encoding of day/year
  - 3/7-day rolling averages for pressure/humidity
  - Weather front detection indicators
- **Advanced Modeling**
  - Stacked ensemble of CatBoost/XGBoost/LightGBM
  - Probability calibration with isotonic regression
  - Dynamic class weighting for imbalance
- **Validation**
  - Time-series aware cross-validation (5 splits)
  - Chronological holdout validation
  - SHAP-based feature selection

### Models Used
| Algorithm | Best Public AUC | Key Features |
|-----------|-----------------|--------------|
| CatBoost | 0.85947 | GPU-accelerated, handle categorical features |
| XGBoost | 0.85411 | Gradient boosting with DART |
| LightGBM | 0.82354 | Efficient histogram-based |
| Random Forest | 0.85572 | Stratified K-Fold validation |
| Neural Network | 0.83319 | TensorFlow implementation |
| Stacked Ensemble | 0.85947 | Logistic Regression meta-model |

---

## Final Submission Results
| Submission File | Public AUC | Private AUC | Approach |
|-----------------|------------|-------------|----------|
| optimized_submission.csv | 0.89823 | 0.85693 | Dynamic Class Weighting CatBoost |
| calibrated_catboost.csv | 0.90312 | 0.84727 | Calibrated probabilities |
| submission_stacked_catboost.csv | 0.88974 | **0.85947** | Best ensemble |
| submission_catboost02_cv.csv | 0.89764 | 0.83829 | Cross-validated CatBoost |
| submission (2).csv | 0.88920 | 0.85411 | Voting classifier |

---

## Challenges Faced
1. **Temporal Overfitting**  
   High variance in time-series patterns between train/test sets

2. **Data Leakage Prevention**  
   Strict management of rolling window features and lagged variables

3. **Class Imbalance**  
   75:25 class ratio required careful weighting and calibration

4. **Computational Constraints**  
   GPU utilization critical for large ensemble training

5. **Model Diversity**  
   Difficulty in creating sufficiently different base learners for stacking

---

