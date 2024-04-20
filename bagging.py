import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

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

# Dopasowanie modeli do danych treningowych
bagging_rf.fit(X_train, y_train)
bagging_gb.fit(X_train, y_train)
bagging_svm.fit(X_train, y_train)
bagging_xgb.fit(X_train, y_train)

# Przewidywanie klas dla danych testowych
y_pred_rf = bagging_rf.predict(X_test)
y_pred_gb = bagging_gb.predict(X_test)
y_pred_svm = bagging_svm.predict(X_test)
y_pred_xgb = bagging_xgb.predict(X_test)

# Obliczenie dokładności modeli
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print("Dokładność modelu Random Forest z bagging:", accuracy_rf)
print("Dokładność modelu Gradient Boosting z bagging:", accuracy_gb)
print("Dokładność modelu SVM z bagging:", accuracy_svm)
print("Dokładność modelu XGBoost z bagging:", accuracy_xgb)