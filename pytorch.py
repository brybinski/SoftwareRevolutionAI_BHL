import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

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
import torch.nn as nn
import torch
import torch.optim as optim

# Definicja modelu autoenkodera adversarialnego
class AutoencoderAdversarial(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoencoderAdversarial, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        self.discriminator = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Przygotowanie danych
# Dane muszą być w formie tensorów PyTorch
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32)

# Parametry modelu
input_dim = X_tensor.shape[1]
encoding_dim = 10

# Inicjalizacja modelu
model = AutoencoderAdversarial(input_dim, encoding_dim)

# Definicja funkcji straty i optymalizatora dla autoenkodera
criterion_ae = nn.MSELoss()
optimizer_ae = optim.Adam(model.parameters(), lr=0.001)

# Trenowanie autoenkodera
num_epochs_ae = 1000
for epoch in range(num_epochs_ae):
    # Przejście przez autoenkoder
    outputs, encoded = model(X_tensor)
    
    # Obliczenie funkcji straty autoenkodera
    loss_ae = criterion_ae(outputs, X_tensor)
    
    # Wsteczna propagacja i aktualizacja wag
    optimizer_ae.zero_grad()
    loss_ae.backward()
    optimizer_ae.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Autoencoder Epoch [{epoch+1}/{num_epochs_ae}], Loss: {loss_ae.item():.4f}')

# Uzyskanie zakodowanych danych z autoenkodera
encoded_data_train = model.encoder(X_tensor).detach().numpy()

# Przygotowanie danych do klasyfikacji Gradient Boosting
# Użyjemy zakodowanych danych jako cech
X_gb_train, X_gb_test, y_gb_train, y_gb_test = train_test_split(encoded_data_train, y_train, test_size=0.2, random_state=42)

# Skalowanie danych
scaler_gb = MinMaxScaler()
X_gb_train_scaled = scaler_gb.fit_transform(X_gb_train)
X_gb_test_scaled = scaler_gb.transform(X_gb_test)

# Utworzenie modelu Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(learning_rate =  0.1, max_depth =  3, min_samples_leaf =  1, min_samples_split = 5, n_estimators = 50)

# Trenowanie modelu Gradient Boosting Classifier
gb_classifier.fit(X_gb_train_scaled, y_gb_train)

# Predykcja na zbiorze testowym
y_gb_pred = gb_classifier.predict(X_gb_test_scaled)

# Obliczenie dokładności klasyfikacji Gradient Boosting
accuracy_gb = accuracy_score(y_gb_test, y_gb_pred)
print("Dokładność klasyfikacji Gradient Boosting:", accuracy_gb)