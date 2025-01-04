# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from SOARegression.transformer import DummyAwareScaler

class SOAR:
    def __init__(self, 
                 regularization=0,
                 scale=True,
                 sample_weight=None
                 ):
        self.base_model = None  # To store the initial linear regression model
        self.coefficients = None  # To store the fitted coefficients
        self.feature_names = None
        self.optimized_coefficients_per_sample = None  # To store optimized coefficients for each sample
        self.regularization = regularization
        self.sample_weight = sample_weight
        self.scale = scale
        
    def fit(self, X, y):
        """
        Fit a linear regression model to the data.
        :param X: Feature matrix (2D numpy array or pandas DataFrame).
        :param y: Target vector (1D numpy array or pandas Series).
        """
        # Initialize the scaler
        if self.scale: 
            self.scaler = DummyAwareScaler()
            
            # Fit the scaler to the data and transform it
            self.scaler.fit(X)
            scaled_X = self.scaler.transform(X)
        else:
            scaled_X = X
        scaled_X = np.column_stack((np.ones(scaled_X.shape[0]), scaled_X))
        self.processed_X = scaled_X
        self.base_model = LinearRegression(fit_intercept=False)
        self.base_model.fit(scaled_X, y, sample_weight=self.sample_weight)
        self.coefficients = self.base_model.coef_
        self.fitted_errors = self.base_model.predict(scaled_X)
        if hasattr(X, "columns"):
            self.feature_names = ["Intercept"] + list(X.columns)
        else:
            self.feature_names = ["Intercept"] + [f"x{i}" for i in range(1, X.shape[1] + 1)]
        # residuals = y - model.predict(X)
        # residual_variance = np.var(residuals, ddof=X.shape[1])
        # design_matrix = X
        # inv_XTX = np.linalg.pinv(design_matrix.T @ design_matrix)  # Use pseudo-inverse
        # self.standard_errors_ = np.sqrt(np.diag(inv_XTX) * residual_variance)

    def optimize_coefficients(self, X, y_targets, column_freeze=None):
        """
        Optimize the regression coefficients for each data point individually.
        :param X: Feature matrix (2D numpy array or pandas DataFrame).
        :param y_targets: Target vector (1D numpy array or pandas Series).
        :column_freeze: A List of column index to not optimize for
        :return: List of optimized coefficients for each sample.
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted before optimization.")
        y_targets = y_targets.copy()
        y_targets = y_targets.astype(np.float64)
        if self.scale: 
            X = self.scaler.transform(X)
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        coefficients = self.coefficients
        if column_freeze is not None: 
            for col in column_freeze: 
                y_targets -= X_with_intercept[:, col] * coefficients[col]
                X_with_intercept = np.delete(X_with_intercept, col, axis=1)
                coefficients = np.delete(coefficients, col)
        self.optimized_coefficients_per_sample = []
        

        for i in range(len(y_targets)):
            try:
                x = X_with_intercept[i]
                y_target = y_targets[i]
    
                # Define the objective function
                def objective(beta):
                    return np.sum((beta - coefficients) ** 2)
    
                # Define the constraint for this sample
                def constraint(beta):
                    return (np.dot(x, beta) - y_target)
    
                # Solve the optimization problem
                constraints = [{'type': 'eq', 'fun': constraint}]
                result = minimize(objective, coefficients, constraints=constraints, method='SLSQP')
    
                if not result.success:
                    raise RuntimeError(f"Optimization failed for sample {i}:", result.message)
    
                # Store the optimized coefficients for this sample
                opt_coefs = result.x
                if self.regularization: 
                    adjustment = self.regularization * (coefficients - opt_coefs)
                    opt_coefs = opt_coefs + adjustment
                self.optimized_coefficients_per_sample.append(opt_coefs)
            except: 
                print(f'Error optimizing for sample: {i}')
                self.optimized_coefficients_per_sample.append(coefficients) 
        self.optimized_coefficients_per_sample = np.asarray(self.optimized_coefficients_per_sample)
        if column_freeze is not None:
            for col in column_freeze: 
                new_col = np.resize(self.coefficients[col], (len(X)))
                self.optimized_coefficients_per_sample = np.insert(
                                                                     self.optimized_coefficients_per_sample,
                                                                     axis=1, 
                                                                     values=new_col, 
                                                                     obj=col
                                                                )

        return self.optimized_coefficients_per_sample
    
    def calc_differences(self):
        return np.log(1 + np.abs(self.coefficients - self.optimized_coefficients_per_sample))
    
    def sample_entropy(self):
        differences = self.calc_differences()
        return np.mean(differences, axis=1)

    def insample_predict(self, X, use_optimized=True):
        """
        Predict using either the optimized coefficients (per sample) or the original coefficients.
        :param X: Feature matrix (2D numpy array or pandas DataFrame).
        :param use_optimized: Whether to use the optimized coefficients (default: True).
        :return: Predicted values (1D numpy array).
        """
        if use_optimized and self.optimized_coefficients_per_sample is None:
            raise ValueError("Optimization must be performed before predicting with optimized coefficients.")

        if self.scale:
            X = self.scaler.transform(X)
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))

        if use_optimized:     
            predictions = np.multiply(X_with_intercept, self.optimized_coefficients_per_sample)
            predictions = np.sum(predictions, axis=1)
        else:     
            predictions = np.dot(X_with_intercept, self.coefficients)
        return predictions

    def plot_coefficients(self, sample_index):
        """
        Plot the original and optimized coefficients for a specific sample.
        :param sample_index: Index of the sample to plot.
        """
        if self.optimized_coefficients_per_sample is None:
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


