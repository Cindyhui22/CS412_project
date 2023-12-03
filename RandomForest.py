# Importing libraries ---------------------------------------------------------- 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve, validation_curve

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'cm' 
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 300

# use a small portion of the dataset for now, remember to remove this line later !!! 
# data_size = 'small'
data_size = 'full'

# Read dataset ----------------------------------------------------------------- 
df = pd.read_csv('Combined_Flights_2018.csv', usecols=['Origin', 'Dest', 'Cancelled', 'CRSDepTime', 'DepDelay', 'CRSElapsedTime', 'Distance', 'Month', 'DayofMonth', 'DayOfWeek', 'IATA_Code_Operating_Airline', 'CRSArrTime', 'ArrDelay', 'ArrivalDelayGroups'])
# Data preprocessing ----------------------------------------------------------- 
# drop cancelled flights 
df_col = df.loc[df['Cancelled'] == False]
df_col = df_col.drop('Cancelled', axis=1)
# copy the dataset
df_raw = df_col.copy() 
# simplify column names 
df_col.rename(columns = {'CRSDepTime': 'DepTime', 'CRSElapsedTime': 'ElapsedTime','DayOfWeek': 'DayofWeek', 'IATA_Code_Operating_Airline': 'Airline', 'CRSArrTime': 'ArrTime', 'ArrivalDelayGroups':'ArrGroup'}, inplace = True)
df_raw.rename(columns = {'CRSDepTime': 'DepTime', 'CRSElapsedTime': 'ElapsedTime','DayOfWeek': 'DayofWeek', 'IATA_Code_Operating_Airline': 'Airline', 'CRSArrTime': 'ArrTime', 'ArrivalDelayGroups':'ArrGroup'}, inplace = True)

### Numerical data 
# Convert 24h-formatted time to minutes 
def time_to_minutes(time_24h):
    """Convert 24h-formatted time to minutes."""
    return (time_24h // 100) * 60 + (time_24h % 100)

df_raw['DepTime'] = df_raw['DepTime'].apply(time_to_minutes)
df_raw['ArrTime'] = df_raw['ArrTime'].apply(time_to_minutes)
# Drop rows with NaN values 
df_raw.dropna(inplace=True) # raw dataset is the dataset without one-hot encoding, this will be used for decision tree regressor  

print(df.columns, df.shape)
print(df_raw.columns, df_raw.shape)

# Split dataset ----------------------------------------------------------------
if data_size == 'small':
    df_raw = df_raw.sample(frac=0.0001, random_state=42)

print(df_raw.columns, df_raw.shape) 

# Split the data into features and target
X = df_raw.drop(columns=['ArrDelay', 'ArrGroup'])
y = df_raw['ArrDelay']

categorical_features = ['Origin', 'Dest', 'Month', 'DayofMonth', 'DayofWeek', 'Airline']

# Apply label encoding to each categorical column
for col in categorical_features: 
    le = LabelEncoder()
    df_raw[col] = le.fit_transform(df_raw[col])

# Now continue with your train-test split and model fitting
X = df_raw.drop(columns=['ArrDelay', 'ArrGroup'])
y = df_raw['ArrDelay']

# Split the data into training and test sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

### Model fitting ---------------------------------------------------------------------------------- 
model_name = 'RandomForest'
# Initialize the RandomForestRegressor --- set the baseline before hyperparameter tuning 
# rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_regressor.fit(X_train, y_train) # Fit the model on the training data
# y_test_pred_rf = rf_regressor.predict(X_test) # Predict on the test set

# # Evaluate the model
# rf_mae = mean_absolute_error(y_test, y_test_pred_rf)
# rf_mse = mean_squared_error(y_test, y_test_pred_rf)
# rf_rmse = np.sqrt(rf_mse)
# rf_r2 = r2_score(y_test, y_test_pred_rf)

# print(f"Random Forest - Mean Absolute Error (MAE): {rf_mae}")
# print(f"Random Forest - Mean Squared Error (MSE): {rf_mse}")
# print(f"Random Forest - Root Mean Squared Error (RMSE): {rf_rmse}")
# print(f"Random Forest - R^2 Score: {rf_r2}")
### ---------------------------------------------------------------------------- 

rf_param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6]
}

# Using GridSearchCV to search for best hyperparameters
rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5, 
                              scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

# Fit the model to training data 
rf_grid_search.fit(X_train, y_train)
print("Best hyperparameters for Random Forest:", rf_grid_search.best_params_)

# Re-train the model with the best hyperparameters from GridSearch 
best_rf_params = rf_grid_search.best_params_
best_rf = RandomForestRegressor(random_state=42, **best_rf_params) 
best_rf.fit(X_train, y_train)

# Evaluate the model with the best hyperparameters
y_test_pred_rf = best_rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, y_test_pred_rf)
rf_mse = mean_squared_error(y_test, y_test_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, y_test_pred_rf)

print(f"Random Forest - Mean Absolute Error (MAE): {rf_mae}")
print(f"Random Forest - Mean Squared Error (MSE): {rf_mse}")
print(f"Random Forest - Root Mean Squared Error (RMSE): {rf_rmse}")
print(f"Random Forest - R^2 Score: {rf_r2}")

### Visualization ---------------------------------------------------------------------------------- 
def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10))
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

def plot_feature_importances(estimator, feature_names, title="Feature Importances"):
    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.bar(range(len(feature_names)), importances[indices], align="center", color='black')
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=60)
    plt.title(title)
    plt.tight_layout()
    return plt

def plot_validation_curve(estimator, X, y, param_name, param_range, title="Validation Curve"):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        scoring="neg_mean_squared_error", n_jobs=-1, cv=5)

    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.figure()
    plt.plot(param_range, train_scores_mean, label="Training score", color="orange")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="green")
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="best")
    return plt

def plot_residuals(y_actual, y_predicted, title="Residuals Plot"):
    residuals = y_actual - y_predicted

    plt.figure()
    plt.scatter(y_predicted, residuals, color='b')
    plt.title(title)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.axhline(y=0, color='r', linestyle='--')
    return plt

def plot_prediction_vs_actual(y_actual, y_predicted, title="Prediction vs Actual Plot"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_predicted, color='b')
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--', lw=2)
    plt.title(title)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    return plt

param_range = np.arange(1, 250, 25)
param_name = "n_estimators"
plot_validation_curve(RandomForestRegressor(), X, y, param_name, param_range, f"Validation Curve - {param_name}").savefig(f"{model_name}_{data_size}_validation_curve.png", bbox_inches='tight', dpi=300)
plot_learning_curve(best_rf, X_train, y_train, title="Learning Curve").savefig(f"{model_name}_{data_size}_learning_curve.png", bbox_inches='tight', dpi=300) 
plot_feature_importances(best_rf, X_train.columns, "Feature Importances").savefig(f"{model_name}_{data_size}_feature_importances.png", bbox_inches='tight', dpi=300)
plot_prediction_vs_actual(y_test, y_test_pred_rf, "Prediction vs Actual").savefig(f"{model_name}_{data_size}_prediction_vs_actual.png", bbox_inches='tight', dpi=300)
plot_residuals(y_test, y_test_pred_rf, "Residuals Plot").savefig(f"{model_name}_{data_size}_residuals.png", bbox_inches='tight', dpi=300)