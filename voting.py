import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Wczytanie danych i podział na zbiór treningowy i testowy


# Inicjalizacja modeli
rf_clf = RandomForestClassifier(random_state=42)
gb_clf = GradientBoostingClassifier(random_state=42)
svm_clf = SVC(kernel='linear', decision_function_shape='ovr', random_state=42)
xgb_clf = XGBClassifier(objective='multi:softmax', num_class=4, random_state=42)

# Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', rf_clf), 
    ('gb', gb_clf), 
    ('svm', svm_clf), 
    ('xgb', xgb_clf)], 
    voting='hard')

# Dopasowanie modelu do danych treningowych
voting_clf.fit(X_train, y_train)

# Przewidywanie klas dla danych testowych
y_pred_voting = voting_clf.predict(X_test)

# Obliczenie dokładności modelu
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print("Dokładność modelu z głosowaniem:", accuracy_voting)