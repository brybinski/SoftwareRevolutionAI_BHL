import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Wczytanie danych
data = pd.read_csv('train.csv')

# Usunięcie brakujących wartości
data = data.dropna().reset_index(drop=True)

# One-Hot Encoding dla kolumny 'Profession'
profession_encoder = OneHotEncoder(sparse=False)
profession_encoded = profession_encoder.fit_transform(data[['Profession']])
profession_df = pd.DataFrame(profession_encoded, columns=profession_encoder.get_feature_names_out(['Profession']))
data = pd.concat([data.drop(['Profession'], axis=1), profession_df], axis=1)

# One-Hot Encoding dla kolumny 'Var_1'
var_1_encoder = OneHotEncoder(sparse=False)
var_1_encoded = var_1_encoder.fit_transform(data[['Var_1']])
var_1_df = pd.DataFrame(var_1_encoded, columns=var_1_encoder.get_feature_names_out(['Var_1']))
data = pd.concat([data.drop(['Var_1'], axis=1), var_1_df], axis=1)

# Label Encoding dla kolumn kategorycznych
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Ever_Married', 'Graduated', 'Spending_Score', 'Segmentation']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Podział danych na zbiór treningowy i testowy
X = data.drop('Segmentation', axis=1)
X = X.drop('ID', axis=1)
y = data['Segmentation']


scaler = MinMaxScaler()
X= scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skalowanie danych


################################################################3
rf_clf = RandomForestClassifier(random_state=42)
gb_clf = GradientBoostingClassifier(random_state=42)
svm_clf = SVC(kernel='linear', decision_function_shape='ovr', random_state=42)
xgb_clf = XGBClassifier(objective='multi:softmax', num_class=4, random_state=42)

# Bagging Classifier dla każdego modelu
bagging_rf = BaggingClassifier(base_estimator=rf_clf, n_estimators=10, random_state=42)
bagging_gb = BaggingClassifier(base_estimator=gb_clf, n_estimators=10, random_state=42)
bagging_svm = BaggingClassifier(base_estimator=svm_clf, n_estimators=10, random_state=42)
bagging_xgb = BaggingClassifier(base_estimator=xgb_clf, n_estimators=10, random_state=42)

# Inicjalizacja Voting Classifier z modelami bazowymi
voting_clf = VotingClassifier(estimators=[
    ('rf', rf_clf), 
    ('gb', gb_clf), 
    ('svm', svm_clf), 
    ('xgb', xgb_clf)], 
    voting='soft')

# Dopasowanie modelu Voting Classifier do danych treningowych
rf_clf = RandomForestClassifier(random_state=42)
gb_clf = GradientBoostingClassifier(random_state=42)
svm_clf = SVC(kernel='linear', decision_function_shape='ovr', probability=True, random_state=42)  # Ustawienie probability=True
xgb_clf = XGBClassifier(objective='multi:softmax', num_class=4, random_state=42)

# Bagging Classifier dla każdego modelu
bagging_rf = BaggingClassifier(base_estimator=rf_clf, n_estimators=10, random_state=42)
bagging_gb = BaggingClassifier(base_estimator=gb_clf, n_estimators=10, random_state=42)
bagging_svm = BaggingClassifier(base_estimator=svm_clf, n_estimators=10, random_state=42)
bagging_xgb = BaggingClassifier(base_estimator=xgb_clf, n_estimators=10, random_state=42)

# Inicjalizacja Voting Classifier z modelami bazowymi
voting_clf = VotingClassifier(estimators=[
    ('rf', rf_clf), 
    ('gb', gb_clf), 
    ('svm', svm_clf), 
    ('xgb', xgb_clf)], 
    voting='soft')

# Dopasowanie modelu Voting Classifier do danych treningowych
voting_clf.fit(X_train, y_train)

# Grid Search dla RandomForest
param_grid_rf = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}
grid_search_rf = GridSearchCV(estimator=rf_clf, param_grid=param_grid_rf, cv=3)
grid_search_rf.fit(X_train, y_train)

# Grid Search dla GradientBoosting
param_grid_gb = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
grid_search_gb = GridSearchCV(estimator=gb_clf, param_grid=param_grid_gb, cv=3)
grid_search_gb.fit(X_train, y_train)

# Grid Search dla SVM
param_grid_svm = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}
grid_search_svm = GridSearchCV(estimator=svm_clf, param_grid=param_grid_svm, cv=3)
grid_search_svm.fit(X_train, y_train)

# Grid Search dla XGBoost
param_grid_xgb = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
grid_search_xgb = GridSearchCV(estimator=xgb_clf, param_grid=param_grid_xgb, cv=3)
grid_search_xgb.fit(X_train, y_train)

# Predykcja dla modelu Voting Classifier
y_pred_voting = voting_clf.predict(X_test)
accuracy_voting = accuracy_score(y_test, y_pred_voting)

# Predykcja i dokładność dla RandomForest z GridSearch
y_pred_rf_grid = grid_search_rf.best_estimator_.predict(X_test)
accuracy_rf_grid = accuracy_score(y_test, y_pred_rf_grid)

# Predykcja i dokładność dla GradientBoosting z GridSearch
y_pred_gb_grid = grid_search_gb.best_estimator_.predict(X_test)
accuracy_gb_grid = accuracy_score(y_test, y_pred_gb_grid)

# Predykcja i dokładność dla SVM z GridSearch
y_pred_svm_grid = grid_search_svm.best_estimator_.predict(X_test)
accuracy_svm_grid = accuracy_score(y_test, y_pred_svm_grid)

# Predykcja i dokładność dla XGBoost z GridSearch
y_pred_xgb_grid = grid_search_xgb.best_estimator_.predict(X_test)
accuracy_xgb_grid = accuracy_score(y_test, y_pred_xgb_grid)

print("Dokładność modelu Voting Classifier (soft voting):", accuracy_voting)
print("Dokładność modelu RandomForest z GridSearch:", accuracy_rf_grid)
print("Dokładność modelu GradientBoosting z GridSearch:", accuracy_gb_grid)
print("Dokładność modelu SVM z GridSearch:", accuracy_svm_grid)
print("Dokładność modelu XGBoost z GridSearch:", accuracy_xgb_grid)