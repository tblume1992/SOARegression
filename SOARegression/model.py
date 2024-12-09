# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class SOAR:
    def __init__(self, 
                 regularization=0):
        self.base_model = None  # To store the initial linear regression model
        self.coefficients = None  # To store the fitted coefficients
        self.feature_names = None
        self.optimized_coefficients_per_sample = []  # To store optimized coefficients for each sample
        self.regularization = regularization
        
    def fit(self, X, y):
        """
        Fit a linear regression model to the data.
        :param X: Feature matrix (2D numpy array or pandas DataFrame).
        :param y: Target vector (1D numpy array or pandas Series).
        """
        self.base_model = LinearRegression()
        self.base_model.fit(X, y)
        self.coefficients = np.append(self.base_model.intercept_, self.base_model.coef_)
        self.fitted_errors = self.base_model.predict(X)
        if hasattr(X, "columns"):
            self.feature_names = ["Intercept"] + list(X.columns)
        else:
            self.feature_names = ["Intercept"] + [f"x{i}" for i in range(1, X.shape[1] + 1)]
        # residuals = y - model.predict(X)
        # residual_variance = np.var(residuals, ddof=X.shape[1])
        # design_matrix = X
        # inv_XTX = np.linalg.pinv(design_matrix.T @ design_matrix)  # Use pseudo-inverse
        # self.standard_errors_ = np.sqrt(np.diag(inv_XTX) * residual_variance)

    def optimize_coefficients(self, X, y_targets):
        """
        Optimize the regression coefficients for each data point individually.
        :param X: Feature matrix (2D numpy array or pandas DataFrame).
        :param y_targets: Target vector (1D numpy array or pandas Series).
        :return: List of optimized coefficients for each sample.
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted before optimization.")
        y_targets = y_targets.copy()
        y_targets = y_targets.astype(np.float64)
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        self.optimized_coefficients_per_sample = []

        for i in range(len(y_targets)):
            x = X_with_intercept[i]
            y_target = y_targets[i]

            # Define the objective function
            def objective(beta):
                return np.sum((beta - self.coefficients) ** 2)

            # Define the constraint for this sample
            def constraint(beta):
                return (np.dot(x, beta) - y_target)

            # Solve the optimization problem
            constraints = [{'type': 'eq', 'fun': constraint}]
            result = minimize(objective, self.coefficients, constraints=constraints, method='SLSQP')

            if not result.success:
                raise RuntimeError(f"Optimization failed for sample {i}:", result.message)

            # Store the optimized coefficients for this sample
            opt_coefs = result.x
            if self.regularization: 
                adjustment = self.regularization * (self.coefficients - opt_coefs)
                opt_coefs = opt_coefs + adjustment
            self.optimized_coefficients_per_sample.append(opt_coefs)

        return self.optimized_coefficients_per_sample

    def predict(self, X, use_optimized=True):
        """
        Predict using either the optimized coefficients (per sample) or the original coefficients.
        :param X: Feature matrix (2D numpy array or pandas DataFrame).
        :param use_optimized: Whether to use the optimized coefficients (default: True).
        :return: Predicted values (1D numpy array).
        """
        if use_optimized and not self.optimized_coefficients_per_sample:
            raise ValueError("Optimization must be performed before predicting with optimized coefficients.")

        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        predictions = []

        for i in range(len(X)):
            if use_optimized:
                beta = self.optimized_coefficients_per_sample[i]
            else:
                beta = self.coefficients
            predictions.append(np.dot(X_with_intercept[i], beta))

        return np.array(predictions)

    def plot_coefficients(self, sample_index):
        """
        Plot the original and optimized coefficients for a specific sample.
        :param sample_index: Index of the sample to plot.
        """
        if not self.optimized_coefficients_per_sample:
            raise ValueError("Optimization must be performed before plotting coefficients.")

        original_coefficients = self.coefficients
        optimized_coefficients = self.optimized_coefficients_per_sample[sample_index]

        indices = np.arange(len(original_coefficients))

        plt.figure(figsize=(10, 6))
        width = 0.35  # Bar width

        # Plot the coefficients
        plt.bar(indices - width / 2, original_coefficients, width, label="Original Coefficients", alpha=0.7)
        plt.bar(indices + width / 2, optimized_coefficients, width, label="Optimized Coefficients", alpha=0.7)

        # Add labels and legend
        plt.xticks(indices, self.feature_names, rotation=45)
        plt.ylabel("Coefficient Value")
        plt.title(f"Comparison of Original and Optimized Coefficients for Sample {sample_index}")
        plt.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()

#%%

if __name__ == "__main__":
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
#%%
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