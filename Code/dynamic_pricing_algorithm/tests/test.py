import numpy as np

categories = {
    "Environmental": ["co2_emissions", "distance", "num_seats"],
    "Socio-economic": ["entry_in_holiday", "entry_in_high_season", "is_resident", "overnight", "visit_hours"],
    "Behavioral": ["cumulative_entries", "likelihood_to_come"]
}

w1 = 2/3
w2 = 1/3
w3 = 0

r1 = 0.5
r2 = 0.4
r3 = 0.9

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


total_sum = sum(weights["Environmental"]) + sum(weights["Socio-economic"]) + sum(weights["Behavioral"])

print(weights["Environmental"])
print(weights["Socio-economic"])
print(weights["Behavioral"])

print(total_sum)
