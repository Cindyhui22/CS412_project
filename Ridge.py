# Importing libraries ---------------------------------------------------------- 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'cm' 
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 300

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
# Drop one level of one-hot encoded columns to avoid multicollinearity 
origin = pd.get_dummies(df_col['Origin'], prefix='origin', drop_first=True)
dest = pd.get_dummies(df_col['Dest'], prefix='dest', drop_first=True)
month = pd.get_dummies(df_col['Month'], prefix='month', drop_first=True)
day_m = pd.get_dummies(df_col['DayofMonth'], prefix='day_m', drop_first=True)
day_w = pd.get_dummies(df_col['DayofWeek'], prefix='day_w', drop_first=True)
airline = pd.get_dummies(df_col['Airline'], prefix='airline', drop_first=True)

# drop original columns
df_col = df_col.drop(columns=['Origin', 'Dest', 'Month', 'DayofMonth', 'DayofWeek', 'Airline'])
# combine dataset together 
df_new = pd.concat([df_col, origin], axis=1)
df_new = pd.concat([df_new, dest], axis=1)
df_new = pd.concat([df_new, month], axis=1)
df_new = pd.concat([df_new, day_m], axis=1)
df_new = pd.concat([df_new, day_w], axis=1)
df_new = pd.concat([df_new, airline], axis=1)

### Numerical data 
# Convert 24h-formatted time to minutes 
def time_to_minutes(time_24h):
    """Convert 24h-formatted time to minutes."""
    return (time_24h // 100) * 60 + (time_24h % 100)

df_new['DepTime'] = df_new['DepTime'].apply(time_to_minutes)
df_new['ArrTime'] = df_new['ArrTime'].apply(time_to_minutes)


'''
We will do the normalization and standardization after splitting the dataset. 
This is to avoid data leakage. 
'''

# Drop rows with NaN values 
df_new.dropna(inplace=True) # new dataset is the dataset with one-hot encoding for categorical variables 
df_raw.dropna(inplace=True) # raw dataset is the dataset without one-hot encoding, this will be used for decision tree regressor  


### Split Data ------------------------------------------------------------------------------------- 
### use a small portion of the dataset for now, be sure to remove this line when training the model !!!!! 
if data_size == 'small': 
    df_new = df_new.sample(frac=0.0001, random_state=42)

# Split the data into features and target
X = df_new.drop(columns=['ArrDelay', 'ArrGroup'])
y = df_new['ArrDelay']

# Split the data into training and test sets (90% train, 10% test) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Separate numerical and categorical columns
numerical_columns = ['DepTime', 'ArrTime', 'DepDelay', 'ElapsedTime', 'Distance']

# Initialize scalers
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

X_train_std = X_train.copy()
X_test_std = X_test.copy()

X_train_std[numerical_columns] = standard_scaler.fit_transform(X_train[numerical_columns])
X_test_std[numerical_columns] = standard_scaler.transform(X_test[numerical_columns])

y_train_std = standard_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_std = standard_scaler.transform(y_test.values.reshape(-1, 1))

### Model training -------------------------------------------------------------------------------------

# Define the model using Ridge
ridge = Ridge()
model_name = 'Ridge'

# You can keep the same range of alphas or adjust it for Ridge
alphas = np.logspace(-3, 2, 20)

# Define hyperparameters for grid search for Ridge
param_grid_ridge = {
    'alpha': alphas  # regularization strength
}

# Set up the grid search for Ridge with standardized data and return_train_score=True
grid_ridge_std = GridSearchCV(ridge, param_grid_ridge, cv=5, return_train_score=True)

# Fit the Ridge model for standardized data
grid_ridge_std.fit(X_train_std, y_train.ravel())  

# Print the best parameters found for Ridge (Standardized) 
print("Best parameters for Ridge (Standardized): ", grid_ridge_std.best_params_)

# Make predictions for standardized data on the test set using the best estimator from Ridge
y_pred_ridge_std = grid_ridge_std.best_estimator_.predict(X_test_std)

# Calculate Mean Squared Error for standardized data on the test set using Ridge
mse_ridge_std = mean_squared_error(y_test, y_pred_ridge_std) 
print("Mean Squared Error for Ridge (Standardized): ", mse_ridge_std)
### Plotting --------------------------------------------------------------------------------------- 
def plot_validation_curve(grid_search, title="Validation Curve"):
    alphas = grid_search.param_grid['alpha']
    mean_train_scores = grid_search.cv_results_['mean_train_score']
    mean_test_scores = grid_search.cv_results_['mean_test_score']

    plt.figure()
    plt.title(title)
    plt.xlabel("Alpha")
    plt.ylabel("Score")
    plt.semilogx(alphas, mean_train_scores, label="Training score", lw=2, color='orange') 
    plt.semilogx(alphas, mean_test_scores, label="Cross-validation score", lw=2, color='g')
    plt.text(0.1, 0.5, f"Best alpha: {grid_search.best_params_['alpha']:.3e}", 
             fontsize=14, transform=plt.gca().transAxes) 
    plt.legend(loc="best")
    return plt


def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    '''
    A learning curve is a graphical representation of how an algorithm's performance changes as a function of the amount of data it's trained on. 
    The graph allows you to understand the relationship between the model's experience (in terms of number of training samples) 
    and its proficiency (usually in terms of accuracy or some error metric). 
    '''   
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

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, color='b')
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.axhline(y=0, color='r', linestyle='-')
    return plt

def plot_predicted_vs_actual(y_test, y_pred):
    '''
    If every predicted value matches the actual value, all points would lie on a 45-degree line, 
    often called the "line of perfect fit" or "identity line". This line usually starts from the origin and has a slope of 1. 
    If the points are scattered widely around this line, it suggests that there are larger discrepancies between the predicted and actual values.
    '''    
    plt.figure()
    plt.scatter(y_test, y_pred, color='b')
    plt.title("Predicted vs Actual")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    return plt

# Plot Validation Curve using GridSearchCV results for Ridge
plot_validation_curve(grid_ridge_std).savefig(f'{model_name}_{data_size}_validation_curve.png', dpi=300, bbox_inches='tight')

# Plot Learning Curve for Ridge (requires separate computation)
plot_learning_curve(grid_ridge_std.best_estimator_, X_train_std, y_train).savefig(f'{model_name}_{data_size}_learning_curve.png', dpi=300, bbox_inches='tight')

# Plot Residuals for Ridge
plot_residuals(y_test, y_pred_ridge_std).savefig(f'{model_name}_{data_size}_residuals.png', dpi=300, bbox_inches='tight')

# Plot Predicted vs Actual for Ridge
plot_predicted_vs_actual(y_test, y_pred_ridge_std).savefig(f'{model_name}_{data_size}_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
