from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score

class RegressionModel:
    def __init__(self, categorical_features, numeric_features):
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.label_encoders = {}
        self.model = None
        self.X_test = None
        self.y_test = None

    def _encode_categorical_features(self, data):
        for feature in self.categorical_features:
            le = LabelEncoder()
            data[feature] = le.fit_transform(data[feature])
            self.label_encoders[feature] = le
        return data

    def _normalize_numeric_features(self, data):
        scaler = StandardScaler()
        data[self.numeric_features] = scaler.fit_transform(data[self.numeric_features])
        return data


    def train_model(self, data):
        data = self._encode_categorical_features(data)
        data = self._normalize_numeric_features(data)

        X = data.drop(['Price'], axis=1)
        y = data['Price']

        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=17)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(self.X_test)

        print("Coeficientes de la Regresión:")
        for feature, coef in zip(X_train.columns, self.model.coef_):
            print(f"{feature}: {coef}")

        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f'Mean Squared Error: {mse}')
        print(f'Mean Absolute Error: {mae}')
        print(f'R^2: {r2}')



    def predict(self, new_data):
        for feature, le in self.label_encoders.items():
            new_data[feature] = le.transform(new_data[feature])

        new_data[self.numeric_features] = StandardScaler().fit_transform(new_data[self.numeric_features])

        prediction = self.model.predict(new_data.drop('matricula', axis=1))
        return prediction[0]

    def plot_results(self, sample_size=500):
        y_pred = self.model.predict(self.X_test)

        sample_indices = random.sample(range(len(self.y_test)), min(sample_size, len(self.y_test)))
        plt.figure(figsize=(10, 6))

        plt.scatter(self.y_test.iloc[sample_indices], y_pred[sample_indices], alpha=0.5, label='Actual vs Predicted Data')

        m, b = np.polyfit(self.y_test, y_pred, 1)
        plt.plot(self.y_test, m * self.y_test + b, color='red', label='Best Fit Line')

        plt.xlabel('Actual Values (€/h)')
        plt.ylabel('Predicted Values (€/h)')
        plt.title('Actual vs Predicted Values')

        plt.legend()
        plt.grid(True)
        plt.show()
