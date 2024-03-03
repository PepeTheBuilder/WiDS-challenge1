import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Load data
train = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')

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

# Encode categorical columns
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = encoder.fit_transform(df[categorical_columns])

# Correlation dictionary

# Selected features based on correlation
correlation = {
    'DiagPeriodL90D': 1.00,
    'patient_age': 0.06,
    'education_bachelors': 0.04,
    'patient_zip3': 0.04,
    'education_less_highschool': -0.04,
    'income_individual_median': 0.03,
    'widowed': -0.03,
    'home_value': 0.03,
    'education_college_or_above': 0.03,
    'income_household_25_to_35': -0.03,
    'health_uninsured': -0.03,
    'labor_force_participation': 0.03,
    'commute_time': -0.03,
    'family_size': -0.03,
    'education_highschool': -0.03,
    'income_household_10_to_15': -0.03,
    'income_household_100_to_150': 0.03,
    'income_household_median': 0.03,
    'income_household_75_to_100': 0.03,
    'poverty': -0.03,
    'income_household_35_to_50': -0.02,
    'rent_median': 0.02,
    'race_black': -0.02,
    'income_household_six_figure': 0.02,
    'self_employed': 0.02,
    'income_household_15_to_20': -0.02,
    'income_household_under_5': -0.02,
    'family_dual_income': 0.02,
    'unemployment_rate': -0.02,
    'age_40s': 0.02,
    'disabled': -0.02,
    'income_household_150_over': 0.02,
    'income_household_20_to_25': -0.02,
    'age_30s': 0.02,
    'education_graduate': 0.02,
    'density': -0.02,
    'bmi': -0.02,
    'income_household_5_to_10': -0.02,
    'education_stem_degree': 0.02,
    'PM25': -0.02,
    'race_white': 0.02,
    'age_10_to_19': -0.02,
    'race_pacific': 0.01,
    'education_some_college': 0.01,
    'limited_english': -0.01,
    'race_multiple': 0.01,
    'age_50s': -0.01,
    'male': 0.01,
    'female': -0.01,
    'age_over_80': -0.01,
    'married': 0.01,
    'age_under_10': -0.01,
    'Ozone': 0.01,
    'population': -0.01,
    'income_household_50_to_75': -0.01,
    'hispanic': -0.01,
    'housing_units': -0.00,
    'race_other': -0.00,
    'age_20s': 0.00,
    'age_70s': -0.00,
    'never_married': -0.00,
    'patient_id': 0.00,
    'race_asian': 0.00,
    'divorced': 0.00,
    'veteran': -0.00,
    'age_60s': -0.00,
    'rent_burden': 0.00,
    'N02': 0.00,
    'home_ownership': 0.00,
    'age_median': -0.00,
    'race_native': 0.00,
    'farmer': 0.00
}
selected_features = [key for key, value in correlation.items() if abs(value) >= 0.02]

# Define columns for training
cols = [
    'breast_cancer_diagnosis_code', 'metastatic_cancer_diagnosis_code', 'patient_zip3', 'patient_age', 'payer_type',
    'patient_state', 'breast_cancer_diagnosis_desc'
] + selected_features

# Separate train and test data
train = df[df['DiagPeriodL90D'] != 2]
test = df[df['DiagPeriodL90D'] == 2].drop(columns=['DiagPeriodL90D'])

# Initialize AUC scores and test predictions
auc_scores = []
test_preds = []

# StratifiedKFold settings
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model parameters (modify as needed)
params = {
    'depth': 10,
    'random_state': 69,
    'eval_metric': 'AUC',
    'verbose': False,
    'loss_function': 'Logloss',
    'learning_rate': 0.005,
    'iterations': 5000,
    'grow_policy': 'Lossguide',
    'l2_leaf_reg': 10,
    'border_count': 128,
    'min_child_samples': 9,
    'leaf_estimation_method': 'Gradient',
    'leaf_estimation_iterations': 16,
    'leaf_estimation_backtracking': 'Armijo',
    'bootstrap_type': 'Poisson',
    'subsample': 0.93,
    'allow_writing_files': False,
    'task_type': 'GPU',
    'od_type': 'IncToDec'
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
    test_preds.append(preds_test)

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
