# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DummyAwareScaler:
    """
    A scaler that scales only numerical columns while leaving dummy (binary) columns unchanged.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.dummy_columns = []
        self.numerical_columns = []

    def fit(self, data):
        """
        Fit the scaler to the numerical columns of the data.

        Parameters:
            data (pd.DataFrame): The input dataset.

        Returns:
            self
        """
        # Identify dummy columns (binary columns with only 0 and 1 as unique values)
        self.dummy_columns = [
            col for col in data.columns if data[col].nunique() == 2 and set(data[col].unique()) <= {0, 1}
        ]
        
        # Identify numerical columns (non-dummy)
        self.numerical_columns = [col for col in data.columns if col not in self.dummy_columns]
        
        # Fit the scaler on the numerical columns
        if self.numerical_columns:  # Fit only if there are numerical columns
            self.scaler.fit(data[self.numerical_columns])
        
        return self

    def transform(self, data):
        """
        Transform the data using the fitted scaler.

        Parameters:
            data (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The transformed dataset.
        """
        if not self.numerical_columns:
            raise ValueError("The scaler has not been fitted with numerical columns.")
        
        # Copy the data to avoid modifying the original dataset
        transformed_data = data.copy()
        
        # Scale only the numerical columns
        if self.numerical_columns:
            transformed_data[self.numerical_columns] = self.scaler.transform(data[self.numerical_columns])
        
        return transformed_data

    def fit_transform(self, data):
        """
        Fit the scaler to the data and then transform it.

        Parameters:
            data (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The transformed dataset.
        """
        self.fit(data)
        return self.transform(data)