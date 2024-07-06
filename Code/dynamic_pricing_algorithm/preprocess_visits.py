import pandas as pd

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path)

    def clean_data(self):
        # Eliminar filas con '#' en 'num_plate'
        self.data = self.data[~self.data['num_plate'].str.contains('#')]
        self.data['distance'] = self.data['distance'].astype(float)

        if self.data['entry_in_holiday'].dtype != bool:
            self.data['entry_in_holiday'] = self.data['entry_in_holiday'].apply(lambda x: True if x.lower() == 'yes' else False)

        if self.data['entry_in_high_season'].dtype != bool:
            self.data['entry_in_high_season'] = self.data['entry_in_high_season'].apply(lambda x: True if x.lower() == 'yes' else False)

        # Imputar valores perdidos en 'num_seats' con valores de 'num_plate'
        self.data['num_seats'] = self.data['num_seats'].fillna(self.data['num_plate'])

        mask_duplicate = self.data.duplicated(subset='num_plate', keep=False)
        unique_values = self.data.loc[mask_duplicate, 'num_seats'].unique()
        for value in unique_values:
            self.data.loc[(self.data['num_plate'] == value) & (self.data['num_seats'].isnull()), 'num_seats'] = value

        # Filtrar solo los valores numéricos en 'num_seats'
        self.data['num_seats'] = pd.to_numeric(self.data['num_seats'], errors='coerce')

        # Imputar valores perdidos en 'num_seats' por la moda.
        for index, row in self.data[self.data['num_seats'].isnull()].iterrows():
            mode_value = self.data[self.data['environmental_distinctive'] == row['environmental_distinctive']]['num_seats'].mode().values
            if len(mode_value) > 0:
                self.data.at[index, 'num_seats'] = mode_value[0]

        self.data['num_seats'] = self.data['num_seats'].astype(int)

        self.data['cumulative_entries'] = self.data['cumulative_entries'].astype(int)

        def assign_town(row):
            if pd.isnull(row['town']):
                if row['postcode'] == 18412:
                    return 'Bubión'
                elif row['postcode'] == 18413:
                    return 'Capileira'
                elif row['postcode'] == 18411:
                    return 'Pampaneira'
                else:
                    return 'other'
            return row['town']


        self.data['town'] = self.data.apply(assign_town, axis=1)
        self.data['co2_emissions'] = pd.to_numeric(self.data['co2_emissions'], errors='coerce')
        self.data['co2_emissions'] = self.data.groupby('environmental_distinctive')['co2_emissions'].transform(lambda x: x.fillna(x.median()))
        self.data['cumulative_entries'] = pd.to_numeric(self.data['cumulative_entries'], errors='coerce').astype('Int64').fillna(1)

        def convert_time_to_hours(time_str):
            parts = time_str.split(' ')

            days = int(parts[0])
            time_part = parts[2]

            h, m, s = map(int, time_part.split(':'))
            total_hours = (days * 24) + h + (m / 60) + (s / 3600)
            return total_hours

        self.data['visit_time_in_hours'] = self.data['visit_time'].apply(convert_time_to_hours)

        #get rid of non tourist vehicles hours between 1h and 2 weeks.
        self.data = self.data[(self.data['visit_time_in_hours'] >= 1) & (self.data['visit_time_in_hours'] <= 336)]

    def get_processed_data(self):
        return self.data
