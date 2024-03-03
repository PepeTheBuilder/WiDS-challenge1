import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Load data
train = pd.read_csv('training.csv')
test = pd.read_csv('testing.csv')

# Drop 'patient_id' column
train.drop(columns=['patient_id'], inplace=True)
test.drop(columns=['patient_id'], inplace=True)

# Define function to handle missing values
def impute_categorical(df, columns):
    """Impute missing values in categorical columns using mode."""
    mode_values = df[columns].mode().iloc[0]
    df[columns] = df[columns].fillna(mode_values)
    return df

def impute_numerical(df, columns):
    """Impute missing values in numerical columns using median."""
    median_values = df[columns].median()
    df[columns] = df[columns].fillna(median_values)
    return df

def handle_missing_values(df):
    """Handle missing values in a DataFrame."""
    # Identify categorical and numerical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(exclude=['object']).columns

    # Impute missing values
    df = impute_categorical(df, categorical_columns)
    df = impute_numerical(df, numerical_cols)

    return df

# Handle missing values in train and test data
train = handle_missing_values(train)
test = handle_missing_values(test)

# Combine train and test data
test['DiagPeriodL90D'] = 2
df = pd.concat([train, test])
# df = df[~df.index.duplicated()]
# df.reset_index(drop=True, inplace=True)

# Encode categorical columns
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = encoder.fit_transform(df[categorical_columns])

# Define columns for training
cols = ['breast_cancer_diagnosis_code', 'metastatic_cancer_diagnosis_code', 'patient_zip3', 'patient_age', 'payer_type',
        'patient_state', 'breast_cancer_diagnosis_desc']

# Separate train and test data
train = df[df['DiagPeriodL90D'] != 2]
test = df[df['DiagPeriodL90D'] == 2].drop(columns=['DiagPeriodL90D'])

# Initialize AUC scores and test predictions
auc_scores = []
test_preds = []  # Change to list

# StratifiedKFold settings
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model parameters
params = {
    'depth': None,                    # Increase tree depth for more complex models
    'random_state': 69,
    'eval_metric': 'AUC',
    'verbose': False,
    'loss_function': 'Logloss',
    'learning_rate': 0.005,         # Lower learning rate for smoother convergence
    'iterations': 5000,            # Increase the number of iterations for better convergence
    'grow_policy': 'Lossguide',# Use SymmetricTree for more balanced trees
    'l2_leaf_reg': 10,              # Regularization strength, adjust as needed
    'border_count': 128,           # Increase border count for more precise splits
    'min_child_samples': 9,       # Decrease minimum child samples for smaller leaf sizes
    'leaf_estimation_method': 'Gradient',  # Use Newton method for leaf value estimation
    'leaf_estimation_iterations': 16,    # Increase leaf estimation iterations for more accurate values
    'leaf_estimation_backtracking': 'Armijo',  # Adjust backtracking strategy
    'bootstrap_type': 'Poisson', # Use Bernoulli bootstrap type for better generalization
    'subsample': 0.93,              # Adjust subsample for more robustness against overfitting
    'allow_writing_files': False,  # Enable lazy processing
    'task_type': 'GPU',             # Specify GPU as the task type for training
    'max_depth': 10,                # Maximum tree depth, can help control model complexity
    'od_type': 'IncToDec',              # Type of early stopping strategy (Iter, IncToDec, IterIncToDec)
    # 'od_wait': 100,                 # Number of iterations to wait for early stopping
    # 'rsm': 0.95,                    # Feature fraction for random selection of features on each iteration
    # 'bagging_temperature': 1,       # Control the randomness of bagging for Bayesian bootstrap type
    # 'one_hot_max_size': 50,         # Maximum number of unique values in categorical features for one-hot encoding
    # 'random_strength': 0.96,        # Magnitude of randomness for feature permutations
    # 'fold_permutation_block': 4,    # Size of the permutation block for the folded OOB method
    # 'nan_mode': 'Min',              # Handling of missing values during split computation
    # # 'leaf_estimation_split': 1,     # Number of splits to be considered when estimating leaf values
    # 'model_shrink_rate': 0.1        # Shrinkage rate for the model updates
}

X = train[cols]
y = train['DiagPeriodL90D']

# Perform cross-validation and predict
for train_indices, test_indices in cv.split(X, y):
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    # Initialize CatBoost classifier
    model = CatBoostClassifier(**params)

    # Train the model
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    # Make predictions on the test set
    preds = model.predict_proba(X_test)[:, 1]
    preds_test = model.predict_proba(test[cols])[:, 1]
    test_preds.append(preds_test)  # Append directly to list

    # Calculate AUC score
    auc_score = roc_auc_score(y_test, preds)
    auc_scores.append(auc_score)
    print(f"AUC Score: {auc_score}")

# Print average AUC score
print(f"Average AUC Score: {np.mean(auc_scores)}")

# Combine predictions from different folds
ensemble_preds = np.mean(test_preds, axis=0)

# Prepare submission file
submission = pd.DataFrame()

# Reset the index of the submission DataFrame
submission = submission.reset_index(drop=True)

# Assign predictions to the 'DiagPeriodL90D' column
submission['DiagPeriodL90D'] = ensemble_preds

# Write first column from test.csv
first_column = test.iloc[:, 0]
submission.insert(0, first_column.name, first_column)

# Save the submission file
submission.to_csv('submission_StratifiedKFold.csv', index=False)