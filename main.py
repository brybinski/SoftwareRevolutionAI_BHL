import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Wczytanie danych
data = pd.read_csv('train.csv')

# Usunięcie brakujących wartości

missing_values_per_column = data.isna().sum()
print(missing_values_per_column)
imp=SimpleImputer(strategy='most_frequent')
list=['Ever_Married','Graduated','Profession','Var_1', 'Work_Experience', 'Family_Size']

for i in list:
    data[i]=imp.fit_transform(data[i].values.reshape(-1,1))


missing_values_per_column = data.isna().sum()
print(missing_values_per_column)
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
# Inicjalizacja i dopasowanie modelu Random Forest
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

# Ocena modelu
y_pred = model_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność modelu Random Forest:", accuracy)




################################################################
import xgboost as xgb
xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=4, random_state=42)
xgb_clf.fit(X_train, y_train)

# Przewidywanie klas dla danych testowych
y_pred = xgb_clf.predict(X_test)

# Obliczenie dokładności modelu
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność modelu XGBoost:", accuracy)



#########################################3

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Inicjalizacja klasyfikatora Gradient Boosting
gb_clf = GradientBoostingClassifier(random_state=42)

# Dopasowanie modelu do danych treningowych
gb_clf.fit(X_train, y_train)

# Przewidywanie klas dla danych testowych
y_pred_gb = gb_clf.predict(X_test)

# Obliczenie dokładności modelu
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print("Dokładność modelu Gradient Boosting:", accuracy_gb)



########################################

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Inicjalizacja klasyfikatora SVM
svm_clf = SVC(kernel='linear', decision_function_shape='ovr', random_state=42)

# Dopasowanie modelu do danych treningowych
svm_clf.fit(X_train, y_train)

# Przewidywanie klas dla danych testowych
y_pred_svm = svm_clf.predict(X_test)

# Obliczenie dokładności modelu
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Dokładność modelu SVM:", accuracy_svm)


######################################################

