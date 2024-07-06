from preprocess_visits import DataPreprocessor
from dynamic_pricing_algorithm import PriceAssigner
from regression_model import RegressionModel
import pandas as pd


def main():
    data_preprocessor = DataPreprocessor('processed_visits.csv')
    data_preprocessor.load_data()
    data_preprocessor.clean_data()
    processed_data = data_preprocessor.get_processed_data()

    processed_data.to_csv('processed_visits.csv', index=False)

    categorical_features = ['likelihood', 'entry_in_holiday', 'entry_in_high_season', 'is_resident']
    numeric_features = ['cumulative_entries','co2_emissions', 'num_seats', 'distance', 'nights', 'visit_time_in_hours']


    
    price_assigner = PriceAssigner('processed_visits.csv', categorical_features, numeric_features)
    price_assigner.load_data()
    processed_prices = price_assigner.assign_prices()

    processed_prices = processed_prices[categorical_features+numeric_features+['Price']]
    processed_prices.to_csv('processed_prices.csv', index=False)


    processed_prices = pd.read_csv('processed_prices.csv')
    regression_model = RegressionModel(categorical_features, numeric_features)
    regression_model.train_model(processed_prices)
    regression_model.plot_results()

if __name__ == "__main__":
    main()
