## Developed By: SAFEEQ FAZIL A
## Register no: 212222240086
## Date:
# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION

### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:

## A - LINEAR TREND ESTIMATION
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = '/content/vegpred.csv' 
dataset = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
dataset['Date'] = pd.to_datetime(dataset['Date'])

# Select a specific commodity to analyze
commodity_name = 'Potato Red' 
commodity_data = dataset[dataset['Commodity'] == commodity_name]

# Sort data by date
commodity_data = commodity_data.sort_values('Date')

# Convert date to numerical values (e.g., days since the first date)
commodity_data['DateNumeric'] = (commodity_data['Date'] - commodity_data['Date'].min()).dt.days

# Independent variable (date as numeric) and dependent variable (average price)
X = commodity_data['DateNumeric'].values.reshape(-1, 1)
y = commodity_data['Average'].values

# Fit a linear regression model
linear_model = LinearRegression()
linear_model.fit(X, y)
linear_trend = linear_model.predict(X)

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(commodity_data['Date'], y, color='blue', label='Actual Prices')
plt.plot(commodity_data['Date'], linear_trend, color='red', label='Linear Trend')
plt.title(f'Linear Trend for {commodity_name}')
plt.xlabel('Date')
plt.ylabel('Average Price (in Rs.)')
plt.legend()
plt.grid(True)
plt.show()


```
## B- POLYNOMIAL TREND ESTIMATION

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
file_path = '/content/vegpred.csv'  
dataset = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
dataset['Date'] = pd.to_datetime(dataset['Date'])

# Select a specific commodity to analyze
commodity_name = 'Potato Red' 
commodity_data = dataset[dataset['Commodity'] == commodity_name]

# Sort data by date
commodity_data = commodity_data.sort_values('Date')

# Convert date to numerical values (e.g., days since the first date)
commodity_data['DateNumeric'] = (commodity_data['Date'] - commodity_data['Date'].min()).dt.days

# Independent variable (date as numeric) and dependent variable (average price)
X = commodity_data['DateNumeric'].values.reshape(-1, 1)
y = commodity_data['Average'].values

# Fit a polynomial regression model (you can change the degree)
degree = 3  # Adjust the degree as needed
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
polynomial_trend = poly_model.predict(X_poly)

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(commodity_data['Date'], y, color='blue', label='Actual Prices')
plt.plot(commodity_data['Date'], polynomial_trend, color='green', label=f'Polynomial Trend (Degree {degree})')
plt.title(f'Polynomial Trend for {commodity_name} (Degree {degree})')
plt.xlabel('Date')
plt.ylabel('Average Price (in Rs.)')
plt.legend()
plt.grid(True)
plt.show()


```

### OUTPUT
## A - LINEAR TREND ESTIMATION
![image](https://github.com/user-attachments/assets/4d004ef3-87a7-4b27-8a77-db51d8ba5777)



## B- POLYNOMIAL TREND ESTIMATION
![image](https://github.com/user-attachments/assets/6dc492e6-5966-41d7-9ca0-31dfb0d79373)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
