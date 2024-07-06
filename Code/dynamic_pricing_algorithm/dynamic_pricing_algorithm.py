import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

class PriceAssigner:
    def __init__(self, data_path, categorical_features, numeric_features):
        self.data_path = data_path
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.categorical_transformer = OneHotEncoder()
        self.numeric_transformer = StandardScaler()

        self.variables = ['cumulative_entries_factor', 'likelihood_factor', 'holiday_factor', 'high_season_factor',
                          'resident_factor', 'km_factor', 'seats_factor',
                          'co2_factor', 'nights_factor', 'visit_time_factor']

        self.environmental_variables = ['co2_factor', 'km_factor', 'seats_factor']
        self.behavioral_variables = ['likelihood_factor', 'cumulative_entries_factor']
        self.socioeconomic_variables = ['holiday_factor', 'high_season_factor', 'resident_factor', 'nights_factor', 'visit_time_factor']

        self.lambda_behavioural = 1/3
        self.lambda_socioeconomic = 1/3
        self.lambda_environmental = 1/3

        self.alpha = 0.5
        self.beta = 1

        self.ratio = 0.9
        self.weights = self.assign_geometric_weights_divided(self.variables, self.ratio)

        print(self.weights)

    def load_data(self):
        self.data = pd.read_csv(self.data_path)

    def preprocess_data(self, data):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numeric_features),
                ('cat', self.categorical_transformer, self.categorical_features)
            ])
        return preprocessor.fit_transform(data)

    def assign_geometric_weights(self, variables, ratio):
        weights = {var: ratio**i for i, var in enumerate(variables)}
        total = sum(weights.values())
        normalized_weights = {var: w / total for var, w in weights.items()}
        return normalized_weights

    def assign_geometric_weights_divided(self, variables, ratio):
        weights = {}

        for category_vars, category_weight in [(self.environmental_variables, self.lambda_environmental),
                                               (self.behavioral_variables, self.lambda_behavioural),
                                               (self.socioeconomic_variables, self.lambda_socioeconomic)]:
            cat_weights = self.assign_geometric_weights(category_vars, ratio)
            for var in category_vars:
                weights[var] = cat_weights[var] * category_weight

        total = sum(weights.values())
        normalized_weights = {var: w / total for var, w in weights.items()}
        return normalized_weights

    def MinMax_normalization(self, value, min_val, max_val, better_high):
        if value < min_val:
            value = min_val
        elif value > max_val:
            value = max_val

        normalized = (value - min_val) / (max_val - min_val)
        return 1 - normalized if better_high else normalized

    def calculate_variable_limits(self, variable_name):
        if variable_name == "num_seats":
            min_limit = 2
            max_limit = 5
        else:
            q1, q3 = np.percentile(self.data[variable_name], [25, 75])
            min_limit = max(0, q1 - 1.5 * (q3 - q1))
            max_limit = q3 + 1.5 * (q3 - q1)

        return min_limit, max_limit

    def assign_price(self, row, variable_limits):
        co2_min, co2_max = variable_limits['co2_emissions']
        seats_min, seats_max = variable_limits['num_seats']
        km_min, km_max = variable_limits['distance']
        nights_min, nights_max = variable_limits['nights']
        visit_time_min, visit_time_max = variable_limits['visit_time_in_hours']
        cumulative_entries_min, cumulative_entries_max = variable_limits['cumulative_entries']

        co2_factor = self.MinMax_normalization(row['co2_emissions'], co2_min, co2_max, better_high=False)
        seats_factor = self.MinMax_normalization(row['num_seats'], seats_min, seats_max, better_high=True)
        km_factor = self.MinMax_normalization(row['distance'], km_min, km_max, better_high=False)
        nights_factor = self.MinMax_normalization(row['nights'], nights_min, nights_max, better_high=True)
        visit_time_factor = self.MinMax_normalization(row['visit_time_in_hours'], visit_time_min, visit_time_max, better_high=True)
        cumulative_entries_factor = self.MinMax_normalization(row['cumulative_entries'], cumulative_entries_min, cumulative_entries_max, better_high=True)

        resident_factor = 1 if row['is_resident'] else 0
        holiday_factor = 1 if row['entry_in_holiday'] else 0
        high_season_factor = 1 if row['entry_in_high_season'] else 0
        likelihood_factor = 1 if row['likelihood'] == 'Vendría más veces' else 0

        categorical_better_high = {
            'resident_factor': False,
            'holiday_factor': True,
            'high_season_factor': True,
            'likelihood_factor': False
        }

        resident_factor = 1 - resident_factor if not categorical_better_high['resident_factor'] else resident_factor
        holiday_factor = holiday_factor if categorical_better_high['holiday_factor'] else 1 - holiday_factor
        high_season_factor = high_season_factor if categorical_better_high['high_season_factor'] else 1 - high_season_factor
        likelihood_factor = likelihood_factor if categorical_better_high['likelihood_factor'] else 1 - likelihood_factor

        weighted_average = (
            likelihood_factor * self.weights['likelihood_factor'] +
            holiday_factor * self.weights['holiday_factor'] +
            high_season_factor * self.weights['high_season_factor'] +
            resident_factor * self.weights['resident_factor'] +
            co2_factor * self.weights['co2_factor'] +
            seats_factor * self.weights['seats_factor'] +
            nights_factor * self.weights['nights_factor'] +
            visit_time_factor * self.weights['visit_time_factor'] +
            km_factor * self.weights['km_factor'] +
            cumulative_entries_factor * self.weights['cumulative_entries_factor']
        )

        price = self.alpha + (self.beta - self.alpha) * weighted_average

        return price

    def assign_prices(self):
        variable_limits = {
            'co2_emissions': self.calculate_variable_limits('co2_emissions'),
            'num_seats': self.calculate_variable_limits('num_seats'),
            'nights': self.calculate_variable_limits('nights'),
            'visit_time_in_hours': self.calculate_variable_limits('visit_time_in_hours'),
            'distance': self.calculate_variable_limits('distance'),
            'cumulative_entries': self.calculate_variable_limits('cumulative_entries')
        }


        self.data['Price'] = self.data.apply(lambda row: self.assign_price(row, variable_limits), axis=1)
        return self.data
