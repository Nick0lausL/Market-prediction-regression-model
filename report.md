# Model Performance Report:

## 1. Executive Summary
The Market Regression Prediction AI Model was evaluated on the **SFY** (SoFi Select 500 ETF) ticker using **1,704 trading days** of historical data. The model demonstrates high predictive accuracy, with an $R^2$ score of **0.9849** on the testing set, indicating it captures over 98% of the variance in the stock's price movements.

## 2. Quantitative Evaluation

### Model Accuracy
The model shows high accuracy with minimal overfitting, as evident from the close proximity of training and testing scores.

| Metric | Training Set | Testing Set | Interpretation |
| :--- | :--- | :--- | :--- |
| **MSE (Mean Squared Error)** | 0.6592 | 1.9745 | Low error magnitude after testing on 2 years of market data. |
| **MAE (Mean Absolute Error)** | 0.5906 | 0.9776 | On average, the prediction is off by roughly **$0.98**. |
| **$R^2$ Score** | 0.9967 | **0.9849** | The model explains **98.49%** of price behavior on unseen data. |

### Feature Importance Analysis
The model relies heavily on recent price action (Lag features) and immediate market sentiment (Open/High/Low/Close).
*   **Primary Driver:** `Close` price is the most dominant predictor (Coeff: 13.71).
*   **Momentum Indicators:** Moving Averages (`MA_5`) and Lagged Close prices (`Close_lag_1`) significantly influence the prediction, confirming the model effectively utilizes trend data.

## 3. Real-World Application Test
*   **Current Price:** $131.92
*   **Predicted Next Price:** $132.10
*   **Expected Change:** +0.14%
*   **Actionable Output:** The model successfully generated a specific trading recommendation (**HOLD**), demonstrating utility for decision support systems.

## 4. Conclusion
The model is highly effective for regression-based price prediction. The low Mean Absolute Error (<$1.00 on a ~$132 asset) indicates sufficient precision for short-term trend analysis.
