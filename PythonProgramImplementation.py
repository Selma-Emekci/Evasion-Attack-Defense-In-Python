import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore
import matplotlib.pyplot as plt

# Function to simulate different types of evasion attacks
def simulate_attack(X, y, attack_type='random_uniform'):
    np.random.seed(1)
    if attack_type == 'random_uniform':
        adversarial_x = np.random.uniform(np.min(X), np.max(X), (10, 1))
    elif attack_type == 'random_normal':
        adversarial_x = np.random.normal(np.mean(X), np.std(X), (10, 1))
    adversarial_y = np.mean(y) + (adversarial_x - np.mean(X)) * 0.8 + np.random.normal(0, np.std(y), (10, 1))
    return adversarial_x, adversarial_y
#Add the file path to your .csv file here
csv_file = 'csv_file.csv'
data = pd.read_csv(csv_file)
X = np.array(data['x'].values).reshape(-1, 1)
y = data['y'].values

# Normalizing the data
X_normalized = (X - np.mean(X)) / np.std(X)

# Fitting the linear regression model to the original data
lr_model = LinearRegression()
lr_model.fit(X_normalized, y)
predictions = lr_model.predict(X_normalized)

# Testing with different attack types
attack_types = ['random_uniform', 'random_normal'] 
for attack_type in attack_types:
    # Simulating attack
    adversarial_x, adversarial_y = simulate_attack(X, y, attack_type)
    adversarial_x_normalized = (adversarial_x - np.mean(X)) / np.std(X)

    # Combining the original and adversarial data
    X_combined_normalized = np.vstack((X_normalized, adversarial_x_normalized))
    Y_combined = np.vstack((y.reshape(-1, 1), adversarial_y.reshape(-1, 1)))

    # Fitting the model with combined data (original + attack)
    attack_model = LinearRegression()
    attack_model.fit(X_combined_normalized, Y_combined)
    predictions_attack = attack_model.predict(X_combined_normalized)

    # Implementing defense mechanism
    z_scores = zscore(Y_combined)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).flatten()

    X_defense = X_combined_normalized[filtered_entries]
    Y_defense = Y_combined[filtered_entries]

    # Fitting the model with defended data
    defense_model = LinearRegression()
    defense_model.fit(X_defense, Y_defense)
    predictions_defense = defense_model.predict(X_defense)

    # Plotting results
    plt.figure(figsize=(7, 4))
    plt.scatter(X_normalized, y, color='blue', label='Original Data')
    plt.plot(X_normalized, predictions, color='red', label='Original Regression Line')
    plt.scatter(adversarial_x_normalized, adversarial_y, color='green', marker='x', label='Adversarial Data')
    plt.plot(X_combined_normalized, predictions_attack, color='purple', label='Regression Line After Attack')
    plt.plot(X_defense, predictions_defense, color='orange', label='Regression Line After Defense')
    plt.title(f'Linear Regression: Original, After {attack_type} Attack, and After Defense')
    plt.xlabel('X (Normalized)')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    # Evaluate MSE
    predictions_on_original_data = defense_model.predict(X_normalized)
    mse = mean_squared_error(predictions, predictions_on_original_data)
    print(f"MSE for {attack_type} attack: {mse}")

