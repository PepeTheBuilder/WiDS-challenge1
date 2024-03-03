import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100)

train = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')
train.drop(columns=['patient_id'],inplace=True)
test.drop(columns=['patient_id'],inplace=True)


numerical_cols = train.select_dtypes(exclude=['object']).columns
categorical_columns = train.select_dtypes(include=['object']).columns

# Impute categorical columns using mode
for col in categorical_columns:
    if col != 'DiagPeriodL90D':
        mode = train[col].mode()[0]
        train[col].fillna(mode, inplace=True)
        test[col].fillna(mode, inplace=True)

# Impute numerical columns using mean
for col in numerical_cols:
    if col != 'DiagPeriodL90D':
        mean = train[col].median()
        train[col].fillna(mean, inplace=True)
        test[col].fillna(mean, inplace=True)

test['DiagPeriodL90D'] = 2
df = pd.concat([train,test])

from sklearn.preprocessing import OrdinalEncoder

# Initialize the encoder
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# Loop through each categorical column
for col in categorical_columns.to_list() + ['patient_zip3']:
    # Fit the encoder on the training data
    encoder.fit(df[[col]])

    # Transform both training and test data
    df[col] = encoder.transform(df[[col]])

cols = ['breast_cancer_diagnosis_code', 'metastatic_cancer_diagnosis_code', 'patient_zip3', 'patient_age', 'payer_type',
        'patient_state', 'breast_cancer_diagnosis_desc']

train = df[df['DiagPeriodL90D'] != 2]
test = df[df['DiagPeriodL90D'] == 2].drop(columns=['DiagPeriodL90D'])

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import numpy as np

X = train[cols + ['DiagPeriodL90D']].drop(columns=['DiagPeriodL90D'], axis=1)
y = train['DiagPeriodL90D']

# Stratejik çapraz doğrulama için katlama ayarları
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model için parametreler
params = {

    'depth': 2,
    'random_state': 42,
    'eval_metric': 'AUC',
    'verbose': False,
    'loss_function': 'Logloss',
    'learning_rate': 0.3,
    'iterations': 1000
}

# AUC skorlarını saklamak için bir liste
auc_scores = []
test_preds = []
for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # CatBoost sınıflandırıcısını başlat
    model = CatBoostClassifier(**params)

    # Modeli eğit
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    # Tahminleri yap
    preds = model.predict_proba(X_test)[:, 1]
    preds_test = model.predict_proba(test[cols])[:, 1]
    test_preds.append(preds_test)
    # AUC skorunu hesapla
    auc_score = roc_auc_score(y_test, preds)
    auc_scores.append(auc_score)
    print(f"AUC Score: {auc_score}")

# Ortalama AUC skorunu yazdır
print(f"Ortalama AUC Skoru: {np.mean(auc_scores)}")
print(pd.DataFrame([1 if prob >= 0.5 else 0 for prob in np.mean(test_preds, axis=0)], columns=['test_preds'])[
          'test_preds'].value_counts())

submission = pd.read_csv('sample_submission.csv')
submission['DiagPeriodL90D'] = np.mean(test_preds,axis=0)
submission.to_csv('submission.csv',index=False)