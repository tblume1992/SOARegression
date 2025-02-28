# SOARegression
Sample Optimized Adaptive Regression

IDK if this is a thing, but I wanted adjustments to coefficients for each sample but in a way that shrinks the difference between the original coefficients and the 'perfect' ones.

To acheive this there is a 2 step process:

1. Fit a basic linear regression and get the coefficients 
2. Do an optimization problem to minimize the squared differences of the coefficients with the constraint that the fitted errors are 0. 

![alt text](https://github.com/tblume1992/SOARegression/blob/main/static/soar_latex.PNG?raw=true "Output 1")


A regularization parameter is introduced to control how tight the fit is which is just a simple shrinkage applied to the coefficients.

## Possible use cases
Time series coefficient weight scheme like a moving average of the last n samples' coefficients 

'Parameter based' outlier detection - you can look at the coefficient swings to find outliers


## Further Enhancements 
I want to add in the coefficient standard errors into the optimization so it is less about absolute magnitude changes and more about minimizing the 'energy'/'entropy' in the system while perfectly fitting.
 
# Pip install
```
pip install soaregression
```

# A Simple Example
```
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from SOARegression.model import SOAR


X = pd.DataFrame([1,2,3,4])
y = pd.Series([3,5,7,10])

model = SOAR(scale=False)
model.fit(X, y)

# Optimize coefficients for each point individually
optimized_coefficients = model.optimize_coefficients(X, y)
#grab normal OLS predictions
predictions = model.insample_predict(X, use_optimized=False)
predictions = pd.Series(predictions, index=X.iloc[:, 0])
#plot each individual sample linear model
for i in range(1,5):
    lin = np.linspace(i-.25, i+.25, 3)
    sample_equation = np.ones(3).reshape(-1, 1) * optimized_coefficients[i-1, 0] + lin.reshape(-1, 1)*optimized_coefficients[i-1, 1]
    plt.plot(pd.Series(sample_equation.reshape(-1), index=lin))
plt.scatter(x=X, y=y, label='Actuals')
plt.plot(predictions, label='OLS')
plt.legend()
plt.show()
```
 ![alt text](https://github.com/tblume1992/SOARegression/blob/main/static/soar_simple.png?raw=true "Output 1")   
After optimizing each sample has it's own linear model that goes through it, but is the most 'similar' to the normal regression line that is fitted. 

Obviously there will be some issues around the intercept but the intercept is fit by default and it the FIRST column in the coefficients.

`print(optimized_coefficients)`

```
[[0.6        2.4       ]
 [0.48       2.26      ]
 [0.46       2.18      ]
 [0.51764706 2.37058824]]
```

# Taking a look at the Airline Passengers with a simple trend + dummy seasonality model
```
import pandas as pd
import numpy as np 

# Load the Air Passengers dataset
data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(data_url, header=0, parse_dates=['Month'], index_col='Month')
df.columns = ['Passengers']

# Extract the time variable
df['Time'] = np.arange(len(df))  # Create a time index for trend

# Extract month from the index
df['Month_Index'] = df.index.month

# One-hot encode the months for seasonality
month_dummies = pd.get_dummies(df['Month_Index'], prefix='Month', drop_first=True)

# Combine time and seasonal features
X = pd.concat([df[['Time']], month_dummies], axis=1) * 1 
y = df['Passengers'].values 

# Instantiate and fit the model
model = SOAR(scale=False)
model.fit(X, y)

# Optimize coefficients for each point individually
optimized_coefficients = model.optimize_coefficients(X, y)
# Predict using the optimized coefficients
predictions = model.insample_predict(X, use_optimized=True)

# Plot coefficients for a specific sample
model.plot_coefficients(sample_index=10)
```
![alt text](https://github.com/tblume1992/SOARegression/blob/main/static/coef_change.png?raw=true "Output 1")

Here we plot the original coefficients vs the optimized ones for a given sample index.

Let's take a look at an optimized vs non-optimize fit:
```
plt.plot(model.insample_predict(X, use_optimized=False), linestyle='dashed', alpha=.5, label='No optimization')
plt.plot(model.insample_predict(X, use_optimized=True), linestyle='dashed', alpha=.5, label='With optimization')
plt.plot(y, alpha=.5, label='Actual')
plt.legend()
plt.show()
```
![alt text](https://github.com/tblume1992/SOARegression/blob/main/static/ap_example.png?raw=true "Output 1")

Optimized will perfectly fit (by design).

# Assessing the sample 'entropy'
```
plt.plot(model.sample_entropy())
```
![alt text](https://github.com/tblume1992/SOARegression/blob/main/static/entropy.png?raw=true "Output 1")

Here entropy isn't a true definition of entropy but it is the best way to describe it. We deal with some intercept endpoint issues but besides that you can see the residuals be reflected in the differences of the cofficients. But just because a residual is large doesn't mean it is difficult to fit it with minimal changes to coefficients.


# Assign custom coefficients based on logic applied to optimized coefficients
Here we will overwrite the optimized coefficients in the class to be equal to the coefficients for the last year, this will give you a fit that works well for those last 12 values.
```
model = SOAR()
model.fit(X, y)

# Optimize coefficients for each point individually
optimized_coefficients = model.optimize_coefficients(X, y)
# Predict using the optimized coefficients
actual_coefs = model.coefficients 
optimized_coefs = model.optimized_coefficients_per_sample
predictions = model.insample_predict(X, use_optimized=True)
mean_coefs = np.resize(optimized_coefs[-12:, :], (len(X), 13))
# overwrite coefficients after getting our weighted average
model.optimized_coefficients_per_sample = mean_coefs
plt.plot(model.insample_predict(X, use_optimized=False), linestyle='dashed', alpha=.5, label='No optimization')
plt.plot(model.insample_predict(X, use_optimized=True), linestyle='dotted', alpha=.5, label='With optimization')
plt.plot(y, alpha=.5, label='Actual')
plt.legend()
plt.show()
```
![alt text](https://github.com/tblume1992/SOARegression/blob/main/static/weighted_example.png?raw=true "Output 1")
# Regularization 
We can apply a basic shrinkage to the coefficients 0 <= regularization <= 1
```
import seaborn as sns 
sns.set_style('darkgrid')
for i in [0,  .5,  1]:
    model = SOAR(regularization=i) 
    model.fit(X, y)

    # Optimize coefficients for each point individually
    optimized_coefficients = model.optimize_coefficients(X, y)

    # Predict using the optimized coefficients
    predictions = model.insample_predict(X, use_optimized=True)
    
    plt.plot(model.insample_predict(X, use_optimized=True), linestyle='dotted', alpha=.9, label=f'regularization={i}')
plt.plot(y, alpha=.5, label='Actual')
plt.legend()
plt.show()
```
![alt text](https://github.com/tblume1992/SOARegression/blob/main/static/regularization.png?raw=true "Output 1")

# A little more complicated of an example
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

# Generate a time index
time = np.arange(0, 200)

# Define trend segments
trend_1 = 0.5 * time[:50]
trend_2 = trend_1[-1] + 0.2 * (time[50:100] - time[50])
trend_3 = trend_2[-1] - 0.3 * (time[100:150] - time[100])
trend_4 = trend_3[-1] + 0.4 * (time[150:200] - time[150])

# Combine the trend segments
trend = np.concatenate([trend_1, trend_2, trend_3, trend_4])

# Add some random noise
noise = np.random.normal(0, 2, size=len(time))

# Create the time series
time_series = trend + noise

# Create a pandas DataFrame for convenience
ts_df = pd.DataFrame({'Time': time, 'Value': time_series})

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(ts_df['Time'], ts_df['Value'], label='Time Series with Changepoints', color='blue')
plt.axvline(x=50, color='red', linestyle='--', label='Changepoint 1')
plt.axvline(x=100, color='green', linestyle='--', label='Changepoint 2')
plt.axvline(x=150, color='orange', linestyle='--', label='Changepoint 3')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Synthetic Time Series with Trend Changepoints')
plt.legend()
plt.show()
```
![alt text](https://github.com/tblume1992/SOARegression/blob/main/static/complicated_series.png?raw=true "Output 1")
# Taking a look at the sample specific models
```
time = pd.DataFrame(time)
time_series = time_series - time_series[0]
model = SOAR(scale=False)
model.fit(time, time_series)
predicted = model.insample_predict(time, use_optimized=False)
# Optimize coefficients for each point individually
optimized_coefficients = model.optimize_coefficients(time, time_series)
entropy = model.sample_entropy()
plt.plot(entropy)
plt.show()
plt.plot(predicted)
plt.plot(time_series)
plt.show() 
for i in range(1,200):
    lin = np.linspace(i-5, i+5, 3)
    sample_equation = np.ones(3).reshape(-1, 1) * optimized_coefficients[i-1, 0] + lin.reshape(-1, 1)*optimized_coefficients[i-1, 1]
    plt.plot(pd.Series(sample_equation.reshape(-1), index=lin))
plt.scatter(x=time, y=time_series, label='Actuals')
plt.plot(predicted, label='OLS', color='black')
plt.xlim(35, 200)
plt.ylim(0, 50)
plt.legend()
plt.show()
```
![alt text](https://github.com/tblume1992/SOARegression/blob/main/static/complicated_universe.png?raw=true "Output 1")
# Constraining what column is optimized 
We can specify NOT to optimize certain columns, here we will constrain the trend to not change which means it will only adjust the intercept term.
```
time = pd.DataFrame(time)
time_series = time_series - time_series[0]
model = SOAR(scale=False)
model.fit(time, time_series)
predicted = model.insample_predict(time, use_optimized=False)
# Optimize coefficients for each point individually
# We will use column_freeze which takes a list of indices to not optimize for
optimized_coefficients = model.optimize_coefficients(time, time_series, column_freeze=[1])
for i in range(1,200):
    lin = np.linspace(i-5, i+5, 3)
    sample_equation = np.ones(3).reshape(-1, 1) * optimized_coefficients[i-1, 0] + lin.reshape(-1, 1)*optimized_coefficients[i-1, 1]
    plt.plot(pd.Series(sample_equation.reshape(-1), index=lin))
plt.scatter(x=time, y=time_series, label='Actuals')
plt.plot(predicted, label='OLS', color='black')
plt.xlim(35, 200)
plt.ylim(0, 50)
plt.legend()
plt.show()
```
![alt text](https://github.com/tblume1992/SOARegression/blob/main/static/frozen_universe.png?raw=true "Output 1")
