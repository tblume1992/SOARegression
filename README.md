# SOARegression
Sample Optimized Adaptive Regression

IDK if this is a thing, but I wanted adjustments to coefficients for each sample but in a way that shrinks the difference between the original coefficients and the 'perfect' ones.

To acheive this there is a 2 step process:

1. Fit a basic linear regression and get the coefficients 
2. Do an optimization problem to minimize the squared differences of the coefficients with the constraint that the fitted errors are 0. 

A regularization parameter is introduced to control how tight the fit is.

## Possible use cases
Time series coefficient weight scheme like a moving average of the last n samples' coefficients 

'Parameter based' outlier detection - you can look at the coefficient swings to find outliers


## Further Enhancements 
I want to add in the coefficient standard errors into the optimization so it is less about absolute magnitude changes and more about minimizing the 'energy'/'entropy' in the system while perfectly fitting.
 
# Code Examples:

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
X = X.values

# Instantiate and fit the model
model = SOAR()
model.fit(X, y)

# Optimize coefficients for each point individually
optimized_coefficients = model.optimize_coefficients(X, y)

# Predict using the optimized coefficients
predictions = model.predict(X, use_optimized=True)

# Print optimized coefficients and predictions
for i, coeffs in enumerate(optimized_coefficients):
    print(f"Optimized Coefficients for Sample {i}:", coeffs)
print("Predictions with Optimized Coefficients:", predictions)

# Plot coefficients for a specific sample
model.plot_coefficients(sample_index=10)

plt.plot(model.predict(X, use_optimized=False), linestyle='dashed', alpha=.5, label='No optimization')
plt.plot(model.predict(X, use_optimized=True), linestyle='dashed', alpha=.5, label='With optimization')
plt.plot(y, alpha=.5, label='Actual')
plt.legend()
plt.show()
```
## Plotting different regularization settings
```
for i in [0, .1, .3, .5, .7, .9, 1]:
    model = SOAR(regularization=i)
    model.fit(X, y)

    # Optimize coefficients for each point individually
    optimized_coefficients = model.optimize_coefficients(X, y)

    # Predict using the optimized coefficients
    predictions = model.predict(X, use_optimized=True)
    
    plt.plot(model.predict(X, use_optimized=True), linestyle='dashed', alpha=.5, label=f'regularization={i}')
    plt.legend()
plt.plot(y, alpha=.9, label='Actual')
plt.show()
```