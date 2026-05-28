# Mini-Project---Restaurant-Revenue-Prediction-Using-Machine-Learning
The objective of this project was to analyze restaurant operational data and build a machine learning model capable of predicting restaurant revenue based on factors such as number of bills, delivery charges, customer count, taxes, and online transactions.


# Restaurant Revenue Prediction Using Machine Learning

## Project Overview

This project focuses on predicting restaurant revenue using Machine Learning techniques based on operational and transactional restaurant data. The objective was to analyze various business factors such as customer bills, delivery charges, taxes, payment modes, and customer count to build a predictive model capable of estimating total restaurant sales revenue accurately.

The project demonstrates practical applications of data preprocessing, exploratory data analysis, feature engineering, regression modeling, leakage detection, and performance evaluation in a real-world business analytics scenario.

---

# Objective

The primary objectives of this project were:

* To analyze restaurant operational and transactional data
* To identify the major factors affecting restaurant revenue
* To build a predictive Machine Learning model for revenue forecasting
* To detect and eliminate data leakage issues
* To evaluate model performance using regression metrics
* To derive business insights from feature importance and correlations

---

# Dataset Used

## Dataset Name

Restaurant Operational and Revenue Dataset

## Dataset Source

Custom restaurant transaction dataset collected from restaurant billing and sales operations.

## Dataset Features

The dataset included features such as:

* Total number of bills
* Delivery charges
* Container charges
* Taxes and GST
* Discounts and waived amounts
* Payment methods (cash, card, UPI, online)
* Customer count (PAX)
* Net sales and total sales revenue

---

# Technologies and Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

---

# Project Workflow

## 1. Data Preprocessing

The following preprocessing steps were performed:

* Removed null rows and empty columns
* Removed duplicate records
* Standardized column names
* Converted currency/string values into numerical format
* Handled missing values using imputation
* Selected only numeric columns for modeling

---

## 2. Exploratory Data Analysis (EDA)

EDA techniques included:

* Correlation heatmap analysis
* Revenue distribution visualization
* Feature correlation analysis
* Detection of highly correlated variables
* Identification of possible data leakage

Key observations:

* Number of bills strongly influenced revenue
* Delivery and container charges positively impacted sales
* Waived amounts negatively affected profitability

---

# Models Used

## Linear Regression

A Linear Regression model was used for revenue prediction because:

* The target variable was continuous numerical data
* Linear Regression provides interpretability through coefficients
* It allows understanding feature impact on revenue
* Suitable for identifying business-driving operational factors

---

# Data Leakage Detection and Removal

Initially, the model produced unrealistically perfect scores:

* R² = 1.0
* Train Score = 1.0
* Test Score = 1.0

This indicated severe data leakage caused by highly correlated financial variables such as:

* cash
* card
* UPI
* online payments
* net sales
* tax-related duplicate variables

These leakage columns were removed to obtain a more realistic and generalizable model.

---

# Final Model Performance

After leakage removal, the model achieved the following results:

| Metric                         | Value          |
| ------------------------------ | -------------- |
| Mean Absolute Error (MAE)      | 94,745         |
| Mean Squared Error (MSE)       | 16,350,211,199 |
| Root Mean Squared Error (RMSE) | 127,867        |
| R² Score                       | 0.987          |

## Cross Validation Scores

```python
[0.9519, 0.9949, 0.9628, 0.6759, 0.9766]
```

### Average Cross Validation Score

```python
0.9124
```

---

# Key Results and Findings

## Strong Revenue Drivers

The model identified the following major positive contributors to restaurant revenue:

* Delivery charges
* Container charges
* Number of bills
* Customer count (PAX)

## Negative Revenue Factors

The following variables negatively affected profitability:

* Waived amounts
* Ecommerce tax deductions
* Online tax calculations

## Operational Insights

The project revealed that:

* Restaurant revenue is heavily driven by delivery operations
* Online ordering significantly contributes to sales
* Higher transaction volume directly increases revenue
* Packaging and takeaway operations are major revenue channels

---

# Feature Importance

Top features influencing revenue prediction:

| Feature               | Coefficient Impact |
| --------------------- | ------------------ |
| delivery_charge       | Strong Positive    |
| container_charge      | Positive           |
| total_no._of_bills    | Positive           |
| round_off             | Slight Positive    |
| waived_off            | Negative           |
| gst_paid_by_ecommerce | Negative           |

---

# Visualization

The project included:

* Correlation Heatmaps
* Revenue Distribution Histograms
* Actual vs Predicted Scatter Plots
* Feature Importance Analysis

These visualizations helped validate model behavior and business trends.

---

# Conclusion

This project successfully demonstrated the application of Machine Learning in restaurant business analytics and revenue forecasting. The final Linear Regression model achieved high predictive accuracy while maintaining realistic generalization after removing leakage features.

The analysis highlighted the importance of delivery operations, customer transactions, and online sales ecosystems in modern restaurant revenue generation.

---

# Future Improvements

Possible future enhancements include:

* Using larger real-world datasets
* Incorporating seasonal and festival trends
* Applying advanced models such as Random Forest and XGBoost
* Deploying the model as a web application/dashboard
* Real-time restaurant sales forecasting

---

# Author

Jose Srambicken

Machine Learning | Data Analytics | Business Intelligence
