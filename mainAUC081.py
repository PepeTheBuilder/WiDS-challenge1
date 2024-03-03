import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Load data
train = pd.read_csv('/kaggle/input/justnumwids/lacrima_training.csv')
test = pd.read_csv('/kaggle/input/justnumwids/lacrima_testing.csv')

# Drop 'patient_id' column
train.drop(columns=['patient_id'], inplace=True)
test.drop(columns=['patient_id'], inplace=True)

# Combine two or more numeric columns into a new column
train['new_column'] = train['patient_age'] + train['patient_zip3'] * 100
test['new_column'] = test['patient_age'] + test['patient_zip3'] * 100  

# Define function to handle missing values
def impute_categorical(df, columns):
    """Impute missing values in categorical columns using mode."""
    for col in columns:
        mode_value = df[col].mode().iloc[0] if not df[col].isnull().all() else np.nan
        df[col] = df[col].fillna(mode_value)
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

# GroupKFold settings
groups = train['new_column']  # Replace 'group_column' with the appropriate column name in your dataset
cv = GroupKFold(n_splits=5)

params = {
    'depth': 5,
    'random_state': 69,
    'eval_metric': 'AUC',
    'verbose': False,
    'loss_function': 'Logloss',
    'learning_rate': 0.05,
    'iterations': 2000,
    'grow_policy': 'Lossguide',
    'l2_leaf_reg': 3,
    'border_count': 254,
    'min_child_samples': 20,
    'leaf_estimation_method': 'Newton',  # Changed to 'Gradient' for CPU
    'leaf_estimation_iterations': 10,
    'bootstrap_type': None,  # Removed for CPU
    'allow_writing_files': False,
    'task_type': 'CPU',
    'od_type': 'Iter',
    'od_wait': 50,
}


X = train[cols]
y = train['DiagPeriodL90D']

# Perform cross-validation and predict
for train_indices, test_indices in cv.split(X, y, groups):
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
submission.to_csv('/kaggle/working/submission_GroupKFold.csv', index=False)
