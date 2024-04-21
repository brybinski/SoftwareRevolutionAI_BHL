import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

# Funkcja do wczytania modelu z pliku pickle
def load_model(model_file):
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    return model

# Ścieżka do pliku z modelem
model_file = 'model_rf.pkl'

# Wczytanie modelu
model = load_model(model_file)

# Dane do przetworzenia (przykładowe dane)
data = pd.read_csv('test.csv')

# Podział danych na cechy i etykiety

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

data = data.drop(['ID'], axis=1)
# Label Encoding dla kolumn kategorycznych
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Ever_Married', 'Graduated', 'Spending_Score']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])


scaler = MinMaxScaler()
X= scaler.fit_transform(data)




# Predykcje na przetworzonych danych
predictions = model.predict(X)

# Wyświetlenie predykcji
print("Predykcje:", predictions)