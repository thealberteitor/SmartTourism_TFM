categories = {
    "Environmental": ["co2_emissions", "distance", "num_seats"],
    "Socio-economic": ["entry_in_holiday", "entry_in_high_season", "is_resident", "overnight", "visit_hours"],
    "Behavioral": ["cumulative_entries", "likelihood_to_come"]
}

w1 = w2 = w3 = 1/3
r1 = r2 = r3 = 0.9



def calculate_weights(category_variables, ratio, category_weight):
    m = len(category_variables)
    weights = [ratio**(k-1) for k in range(1, m+1)]
    total = sum(weights)
    normalized_weights = [w / total * category_weight for w in weights]
    return normalized_weights


weights = {}
weights["Environmental"] = calculate_weights(categories["Environmental"], r1, w1)
weights["Socio-economic"] = calculate_weights(categories["Socio-economic"], r2, w2)
weights["Behavioral"] = calculate_weights(categories["Behavioral"], r3, w3)


sum_environmental = sum(weights["Environmental"])
sum_socioeconomic = sum(weights["Socio-economic"])
sum_behavioral = sum(weights["Behavioral"])


total_sum = sum_environmental + sum_socioeconomic + sum_behavioral
sum_environmental, sum_socioeconomic, sum_behavioral, total_sum
