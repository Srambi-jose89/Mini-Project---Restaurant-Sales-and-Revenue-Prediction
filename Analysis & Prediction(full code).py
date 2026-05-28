import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_excel('/content/drive/MyDrive/ML_Project/restaurant_data.xlsx')

df=df.dropna(how='all')
df=df.dropna(axis=1, how='all')
df=df.drop_duplicates()

df.shape

df.columns = df.columns.str.strip()
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ', '_')

df.columns

df.isnull().sum()

df.fillna(0, inplace=True)

df = df.drop(columns=['source_file'])
df = df.dropna(axis=1, thresh=len(df)*0.5)

df.head(10)


df.dtypes

for col in df.columns:
  #convert everything to string first
  df[col] = df[col].astype(str)

  #Remove commas, currency symbols and spaces
  df[col]=df[col].str.replace(',', '', regex=False)
  df[col]=df[col].str.replace('₹', '', regex=False)
  df[col]=df[col].str.strip()

  #Apply to_numeric for each column inside the loop, coercing errors to NaN
  df[col]=pd.to_numeric(df[col], errors='coerce')

#Fill NaNs after numeric conversion to handle coerced values
df.fillna(0, inplace=True)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# print(numeric_cols)

# print(df.columns)

import matplotlib.pyplot as plt
import seaborn as sns

#Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


#Histogram
plt.figure(figsize=(8,5))
sns.histplot(df['total_sales_(₹)'], bins=20, kde=True)
plt.title("Revenue Distribution")
plt.xlabel("Revenue")
plt.ylabel("Frequency")
plt.show()

#Feature Selection

target = 'total_sales_(₹)'

X = df.drop(columns=[target])
y = df[target]

#Remove non-numeric columns

X = X.select_dtypes(include=['int64', 'float64'])

X.columns

#Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)
#Linear Regression Model

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#Predictions

y_pred = model.predict(X_test)

#Model Evaluation

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

#The Data has Data leakage as a cause for Accuracy in Prediction as well as it can be overfitting

print(X.columns)

X = X.drop(columns=[
    'revenue',
    'sales_copy',
    'profit',
    'net_sales_(₹)'
], errors='ignore')

print(X.T.duplicated())

corr = df.corr(numeric_only=True)

print(corr['net_sales_(₹)(m.a_-_t.d)'].sort_values(ascending=False))

print(df.duplicated().sum())

df = df.drop_duplicates()

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(train_score)
print(test_score)

#As the value returned is train =1.0 and test =1.0 moving on with the assumption that there is possible data leakage

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print(coef_df)

#this also says there is severe data leakage

#To remove leakage columns

leakage_cols = [
    'cash',
    'card',
    'upi',
    'online',
    'other',
    'not_paid',
    'wallet',
    'due_payment',
    'net_sales_(₹)(m.a_-_t.d)',
    'my_amount_(₹)',
    'total_discount_(₹)',
    'total_tax_(₹)'
]

X = X.drop(columns=leakage_cols, errors='ignore')

print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

corr = df[X.columns.tolist() + [target]].corr(numeric_only=True)

print(corr[target].sort_values(ascending=False))

#Mean Absolute Error: 94745.28 Mean Square Error: 16350211199.33 Root Mean Square Error: 127867.94 R²: 0.9871 actually indicate: Very strong model performance, Leakage greatly reduced, Realistic prediction behavior. This is MUCH healthier than your earlier fake-perfect results.

#What this suggests Your dataset probably still has: Strong predictive patterns but possibly some remaining highly correlated features.

#Cheching remaining correlations
corr = df.corr(numeric_only=True)

print(corr[target].sort_values(ascending=False))


#To check coeffecients Again
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print(coef_df)


#cross validation to make model more realistic healthier, free from major leakage etc
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print(scores)
print(scores.mean())

#Compare Actual Value vs Predicted 

results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

results.head(10)


#Visualization of Predictions

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')

plt.show()

#If points are near a straight line: model is performing well.


#Feature Importance
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
coefficients.sort_values(by='Coefficient', ascending=False)

