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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Wykres "łokcia" - pomocny w doborze optymalnej liczby klastrów
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Liczba klastrów')
plt.ylabel('Suma kwadratów odległości')
plt.title('Metoda "łokcia"')
plt.show()

# Inicjalizacja i dopasowanie modelu k-średnich
num_clusters = 3  # Ustawienie liczby klastrów
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Przewidywanie przynależności do klastrów dla każdego klienta
labels = kmeans.predict(X)

# Dodanie etykiet klastrów do danych
data['Cluster'] = labels

# Wyświetlenie wyników
print(data.head())