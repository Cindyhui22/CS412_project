# Importing libraries ---------------------------------------------------------- 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Initialize the DecisionTreeRegressor
model_name = 'DecisionTree' 
dt_regressor = DecisionTreeRegressor(random_state=42)

# Fit the model on the training data
dt_regressor.fit(X_train, y_train)

# Initialize a Decision Tree Regressor for GridSearch
tree = DecisionTreeRegressor(random_state=42)

# Specify hyperparameters and their possible values
param_grid = {
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 3, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10] 
}

# Using GridSearchCV to search for best hyperparameters
grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='neg_mean_squared_error', 
                           verbose=1, n_jobs=-1, return_train_score=True)

# Fit the model to training data
grid_search.fit(X_train, y_train)

# Best hyperparameters from grid search
best_hyperparams = grid_search.best_params_
print("Best hyperparameters from grid search:", best_hyperparams)

# Get the cost complexity pruning path for the best estimator from grid search
path = grid_search.best_estimator_.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Initialize an array to store the results of each fold during cross-validation
cv_mse_scores = []

# Perform cross-validation for each alpha in ccp_alphas
for ccp_alpha in ccp_alphas:
    dt_regressor = DecisionTreeRegressor(random_state=42, ccp_alpha=ccp_alpha, **best_hyperparams)
    scores = cross_val_score(dt_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_mse_scores.append(-scores.mean())

# Find the best alpha and corresponding minimum MSE
best_alpha_index = np.argmin(cv_mse_scores)
best_alpha = ccp_alphas[best_alpha_index]
min_mse = cv_mse_scores[best_alpha_index]
print(f"Best alpha: {best_alpha} with MSE: {min_mse}")

# Update best_hyperparams with the best alpha
best_hyperparams['ccp_alpha'] = best_alpha

# Re-train the decision tree with the best combination of hyperparameters
best_tree = DecisionTreeRegressor(random_state=42, **best_hyperparams)
best_tree.fit(X_train, y_train)

# Predict on test data using the best model from GridSearch
y_test_pred = best_tree.predict(X_test)

# Metrics for test data
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nTest MSE: {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R^2: {test_r2:.4f}")

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
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt 

def plot_ccp_path(tree, X_train, y_train, model_name, best_alpha):
    path = tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    plt.figure(figsize=(10, 6))
    plt.plot(ccp_alphas[:-1], impurities[:-1], color='b', marker='o', drawstyle="steps-post")
    plt.text(0.1, 0.5, f"Best alpha: {best_alpha:.3f}", 
         fontsize=14, transform=plt.gca().transAxes)
    plt.xlabel("Effective alpha")
    plt.ylabel("Total impurity of leaves")
    plt.title("Cost Complexity Pruning Path")
    return plt 

# Convert DataFrame column names to a list for the feature_names parameter
feature_names = X_train.columns.tolist() 
def plot_decision_tree(tree, feature_names, model_name):
    plt.figure(figsize=(20, 10))
    plot_tree(tree, filled=True, feature_names=feature_names, max_depth=3)
    plt.savefig(f"{model_name}_tree.png", bbox_inches='tight', dpi=300)
    return plt 

### Model evaluation ### 
def plot_residuals(y_test, y_pred, model_name):
    residuals = y_test - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals, color='b')
    plt.title("Residuals Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.axhline(y=0, color='r', linestyle='-')
    return plt 

def plot_predictions_vs_actual(y_test, y_pred, model_name):
    plt.figure()
    plt.scatter(y_test, y_pred, color='b')
    plt.title("Prediction vs Actual")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    return plt

def plot_feature_importances(model, feature_names, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
            color="k", align="center")
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=60)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    return plt

plot_decision_tree(best_tree, feature_names, model_name).savefig(f"{model_name}_{data_size}_tree.png", bbox_inches='tight', dpi=300)
plot_ccp_path(best_tree, X_train, y_train, model_name, best_alpha).savefig(f"{model_name}_{data_size}_ccp_path.png", bbox_inches='tight', dpi=300)
plot_learning_curve(best_tree, X_train, y_train).savefig(f"{model_name}_{data_size}_learning_curve.png", bbox_inches='tight', dpi=300) 
plot_residuals(y_test, y_test_pred, model_name).savefig(f"{model_name}_{data_size}_residuals.png", bbox_inches='tight', dpi=300)
plot_predictions_vs_actual(y_test, y_test_pred, model_name).savefig(f"{model_name}_{data_size}_predictions_vs_actual.png", bbox_inches='tight', dpi=300)
plot_feature_importances(best_tree, feature_names, model_name).savefig(f"{model_name}_{data_size}_feature_importances.png", bbox_inches='tight', dpi=300)