from sklearn.datasets import make_moons
from sklearn.datasets import load_digits

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import numpy as np

import warnings
filterWarnings = True
if(filterWarnings):
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Generate synthetic moon-shaped dataset
moonsX, moonsY = make_moons(n_samples=4500, noise=0.1, random_state=42)
digits = load_digits(n_class=10)
digits_data = digits.data
digits_target = digits.target

# Split the dataset into train and test sets
moonsX_train, moonsX_test, moonsY_train, moonsY_test = train_test_split(moonsX, moonsY, test_size=0.2, random_state=42)
digitsX_train, digitsX_test, digitsY_train, digitsY_test = train_test_split(digits_data, digits_target, test_size=0.2, random_state=42)


print("MOONS")

# Create an instance of LogisticRegression
model = LogisticRegression()

# Fit the model to the training data
model.fit(moonsX_train, moonsY_train)

# Predict the target values for the training and test data
moonsY_train_pred = model.predict(moonsX_train)
moonsY_test_pred = model.predict(moonsX_test)

# Calculate the posterior probabilities for the test data
probabilities = model.predict_proba(moonsX_test)

# Print the posterior probabilities for the first few samples in the test data
for i in range(5):
    print(f"Sample {i+1}: {probabilities[i]}")

# Calculate R2 score
moons_r2_train = r2_score(moonsY_train, moonsY_train_pred)
moons_r2_test = r2_score(moonsY_test, moonsY_test_pred)

# Calculate root mean squared error (RMSE)
moons_rmse_train = np.sqrt(mean_squared_error(moonsY_train, moonsY_train_pred))
moons_rmse_test = np.sqrt(mean_squared_error(moonsY_test, moonsY_test_pred))

# Calculate mean absolute error (MAE)
moons_mae_train = mean_absolute_error(moonsY_train, moonsY_train_pred)
moons_mae_test = mean_absolute_error(moonsY_test, moonsY_test_pred)

# Calculate mean absolute percentage error (MAPE)
moons_mape_train = np.mean(np.abs((moonsY_train - moonsY_train_pred) / moonsY_train)) * 100
moons_mape_test = np.mean(np.abs((moonsY_test - moonsY_test_pred) / moonsY_test)) * 100

# Print the metrics
print("R2 score - Training set:", moons_r2_train)
print("R2 score - Test set:", moons_r2_test)
print("RMSE - Training set:", moons_rmse_train)
print("RMSE - Test set:", moons_rmse_test)
print("MAE - Training set:", moons_mae_train)
print("MAE - Test set:", moons_mae_test)
print("MAPE - Training set:", moons_mape_train)
print("MAPE - Test set:", moons_mape_test)

#                                            GRID START

model = LogisticRegression()

param_grid = {
    'C': [0.1, 1, 10],
    'multi_class':['multinomial','auto'],
    'solver': ['lbfgs', 'saga'],
    'penalty': ['none', 'l2']
}

# Create GridSearchCV instance
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Fit the model to the training data
grid_search.fit(moonsX_train, moonsY_train)

print("GRID SEARCH")
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

model = LogisticRegression(multi_class=grid_search.best_params_['multi_class'], solver=grid_search.best_params_['solver'])

model.fit(moonsX_train, moonsY_train)

moonsY_train_pred = model.predict(moonsX_train)
moonsY_test_pred = model.predict(moonsX_test)

probabilities = model.predict_proba(moonsX_test)

for i in range(5):
    print(f"Sample {i+1}: {probabilities[i]}")

moons_r2_train = r2_score(moonsY_train, moonsY_train_pred)
moons_r2_test = r2_score(moonsY_test, moonsY_test_pred)

# Calculate root mean squared error (RMSE)
moons_rmse_train = np.sqrt(mean_squared_error(moonsY_train, moonsY_train_pred))
moons_rmse_test = np.sqrt(mean_squared_error(moonsY_test, moonsY_test_pred))

# Calculate mean absolute error (MAE)
moons_mae_train = mean_absolute_error(moonsY_train, moonsY_train_pred)
moons_mae_test = mean_absolute_error(moonsY_test, moonsY_test_pred)

# Calculate mean absolute percentage error (MAPE)
moons_mape_train = np.mean(np.abs((moonsY_train - moonsY_train_pred) / moonsY_train)) * 100
moons_mape_test = np.mean(np.abs((moonsY_test - moonsY_test_pred) / moonsY_test)) * 100


# Print the metrics
print("R2 score - Training set:", moons_r2_train)
print("R2 score - Test set:", moons_r2_test)
print("RMSE - Training set:", moons_rmse_train)
print("RMSE - Test set:", moons_rmse_test)
print("MAE - Training set:", moons_mae_train)
print("MAE - Test set:", moons_mae_test)
print("MAPE - Training set:", moons_mape_train)
print("MAPE - Test set:", moons_mape_test)
#                                            GRID END

# DIGITS START

print("DIGITS")

model = LogisticRegression()

model.fit(digitsX_train, digitsY_train)

digitsY_train_pred = model.predict(digitsX_train)
digitsY_test_pred = model.predict(digitsX_test)

probabilities = model.predict_proba(digitsX_test)

for i in range(3):
    print(f"Sample {i+1}: {probabilities[i]}")

digits_r2_train = r2_score(digitsY_train, digitsY_train_pred)
digits_r2_test = r2_score(digitsY_test, digitsY_test_pred)

# Calculate root mean squared error (RMSE)
digits_rmse_train = np.sqrt(mean_squared_error(digitsY_train, digitsY_train_pred))
digits_rmse_test = np.sqrt(mean_squared_error(digitsY_test, digitsY_test_pred))

# Calculate mean absolute error (MAE)
digits_mae_train = mean_absolute_error(digitsY_train, digitsY_train_pred)
digits_mae_test = mean_absolute_error(digitsY_test, digitsY_test_pred)

# Calculate mean absolute percentage error (MAPE)
digits_mape_train = np.mean(np.abs((digitsY_train - digitsY_train_pred) / digitsY_train)) * 100
digits_mape_test = np.mean(np.abs((digitsY_test - digitsY_test_pred) / digitsY_test)) * 100

# Print the metrics
print("R2 score - Training set:", digits_r2_train)
print("R2 score - Test set:", digits_r2_test)
print("RMSE - Training set:", digits_rmse_train)
print("RMSE - Test set:", digits_rmse_test)
print("MAE - Training set:", digits_mae_train)
print("MAE - Test set:", digits_mae_test)
print("MAPE - Training set:", digits_mape_train)
print("MAPE - Test set:", digits_mape_test)

#                                            GRID START
model = LogisticRegression()

param_grid = {
    'C': [0.1, 1, 10],
    'multi_class':['multinomial','auto'],
    'solver': ['lbfgs', 'saga'],
    'penalty': ['none', 'l2']
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

grid_search.fit(digitsX_train, digitsY_train)

print("GRID SEARCH")
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

model = LogisticRegression(multi_class=grid_search.best_params_['multi_class'], solver=grid_search.best_params_['solver'])

model.fit(digitsX_train, digitsY_train)

digitsY_train_pred = model.predict(digitsX_train)
digitsY_test_pred = model.predict(digitsX_test)

probabilities = model.predict_proba(digitsX_test)

for i in range(3):
    print(f"Sample {i+1}: {probabilities[i]}")

digits_r2_train = r2_score(digitsY_train, digitsY_train_pred)
digits_r2_test = r2_score(digitsY_test, digitsY_test_pred)

# Calculate root mean squared error (RMSE)
digits_rmse_train = np.sqrt(mean_squared_error(digitsY_train, digitsY_train_pred))
digits_rmse_test = np.sqrt(mean_squared_error(digitsY_test, digitsY_test_pred))

# Calculate mean absolute error (MAE)
digits_mae_train = mean_absolute_error(digitsY_train, digitsY_train_pred)
digits_mae_test = mean_absolute_error(digitsY_test, digitsY_test_pred)

# Calculate mean absolute percentage error (MAPE)
digits_mape_train = np.mean(np.abs((digitsY_train - digitsY_train_pred) / digitsY_train)) * 100
digits_mape_test = np.mean(np.abs((digitsY_test - digitsY_test_pred) / digitsY_test)) * 100


# Print the metrics
print("R2 score - Training set:", digits_r2_train)
print("R2 score - Test set:", digits_r2_test)
print("RMSE - Training set:", digits_rmse_train)
print("RMSE - Test set:", digits_rmse_test)
print("MAE - Training set:", digits_mae_train)
print("MAE - Test set:", digits_mae_test)
print("MAPE - Training set:", digits_mape_train)
print("MAPE - Test set:", digits_mape_test)
#                                            GRID END

# DIGITS END
