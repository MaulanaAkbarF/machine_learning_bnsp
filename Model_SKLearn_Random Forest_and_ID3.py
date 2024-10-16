import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import os

print(f'--- Algorithm Model: Random Forest with ID3 ---')
current_dir = os.getcwd()
df = pd.read_excel(os.path.join(current_dir, "DataTraining.xlsx"))

le_gender = LabelEncoder()
le_learner = LabelEncoder()

gender_mapping = {
    "Male": 0,
    "Female": 1
}

learner_mapping = {
    "V": 0,
    "A": 1,
    "K": 2
}

le_gender.classes_ = list(gender_mapping.keys())
le_learner.classes_ = list(learner_mapping.keys())

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Learner'] = le_learner.fit_transform(df['Learner'])

if not os.path.exists(os.path.join('Machine_Learning', 'Models')):
    os.makedirs(os.path.join('Machine_Learning', 'Models'))

with open(os.path.join('Machine_Learning', 'Models', 'Encoder_gender.pkl'), 'wb') as f:
    pickle.dump(le_gender, f)

with open(os.path.join('Machine_Learning', 'Models', 'Encoder_learner.pkl'), 'wb') as f:
    pickle.dump(le_learner, f)

print(f'Running: Feature Selection')
features = df.columns.drop(['Gender', 'Age', 'Learner'])
X = df[['Gender', 'Age'] + list(features)]
y = df['Learner']

selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)
X_new = pd.DataFrame(X_new, columns=selector.get_feature_names_out(X.columns))

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25, random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, 9, 11, 13, 15, None],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'class_weight': [None, 'balanced'],
    'criterion': ['entropy'], # deklarasikan dan pake entropy disini
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,
    cv=skf,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42
)

print(f'Running: Hyperparameter Tuning with RandomizedSearchCV')
random_search.fit(X_train_subset, y_train_subset)

best_model = random_search.best_estimator_

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print(f'--- Pelatihan Selesai! ---')

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

recall = recall_score(y_test, y_pred, average='weighted')
print(f'Recall: {recall*100}%')

precision = precision_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision*100}%')

f1 = f1_score(y_test, y_pred, average='weighted')
print(f'F1 Score: {f1*100}%')

accuracy = best_model.score(X_test, y_test)
print(f'Accuracy: {accuracy*100}%')

print(f'Running: 10-fold Cross-Validation')
cv_scores = cross_val_score(
    best_model, 
    X_new, 
    y, 
    cv=10,
    scoring='f1_weighted'
)

print('10-fold Cross-validation scores: ', cv_scores)
print('Average 10-fold Cross-validation score: ', np.mean(cv_scores))

with open(os.path.join('Models', 'Model_SKLearn_RandomForest_ID3.pkl'), "wb") as f:
    pickle.dump(best_model, f)
