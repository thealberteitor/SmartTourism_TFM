import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np


class DataPreprocessor:
    def __init__(self, columns, dataframe, to_predict, categorical_encoding='label', random_state=17):
        self.df = dataframe
        self.random_state = random_state
        self.categorical_encoding = categorical_encoding
        self.label = to_predict
        self.y = self.encode_label()
        self.X = self.df[columns].copy()

        self.categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns
        self.numerical_columns = self.X.select_dtypes(exclude=['object', 'category']).columns


        self.preprocess_data()

    """

    class DataPreprocessor:
        def __init__(self, columns, file_path, categorical_encoding='onehot', random_state=17,  to_predict='park_price_will_affect_behaviour'):
            self.df = pd.read_csv(file_path)
            self.random_state = random_state
            self.categorical_encoding = categorical_encoding
            self.label = to_predict
            self.y = self.encode_label()
            self.X = self.df[columns].copy()

            self.categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns
            self.numerical_columns = self.X.select_dtypes(exclude=['object', 'category']).columns
    """

    def encode_label(self):
        encoder = LabelEncoder()
        return encoder.fit_transform(self.df[self.label])

    def preprocess_data(self):

        #if self.label == 'park_price_will_affect_behaviour':

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore') if self.categorical_encoding == 'onehot' else OrdinalEncoder())
        ])
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        num_cols = self.numerical_columns.tolist() + (['target_label'] if 'target_label' in self.X.columns else [])

        preprocessor = ColumnTransformer([
            ('num', numerical_transformer, num_cols),
            ('cat', categorical_transformer, self.categorical_columns)
        ])

        X_processed = preprocessor.fit_transform(self.X)

        if self.categorical_encoding == 'onehot':
            cat_features = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(self.categorical_columns)
            all_features = num_cols + cat_features.tolist()
        else:
            all_features = num_cols + self.categorical_columns.tolist()

        self.X_processed = pd.DataFrame(X_processed, columns=all_features)
        return self.X_processed, self.y
