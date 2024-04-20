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

# Skalowanie danych


################################################################3 grid search ################################################################
gb_clf = GradientBoostingClassifier()

# Definicja siatki parametrów do przeszukiwania
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.1, 0.5],
#     'max_depth': [3, 5, 7]
# }
param_grid = {}
# Inicjalizacja GridSearchCV
grid_search = GridSearchCV(estimator=gb_clf, param_grid=param_grid, cv=3, scoring='accuracy')

# Dopasowanie modelu do danych treningowych
grid_search.fit(X_train, y_train)

# Wybór najlepszego modelu
best_gb_clf = grid_search.best_estimator_

# Predykcja dla danych testowych
y_pred = best_gb_clf.predict(X_test)
#########################################################  accuracy ###############################
# Obliczenie dokładności modelu
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność modelu Gradient Boosting Classifier:", accuracy)

# Wyświetlenie najlepszych parametrów
print("Najlepsze parametry:", grid_search.best_params_)

######################################################### CONFUSION MATRIX ###############################
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Macierz pomyłek')
plt.colorbar()
classes = ['Klasa 0', 'Klasa 1', 'Klasa 2', 'Klasa 3']  # Etykiety klas
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('Prawdziwa klasa')
plt.xlabel('Przewidziana klasa')
plt.show()


######################################################### PRECISION ###############################
from sklearn.metrics import precision_score

# Prawdziwe etykiety (y_true) i przewidywane etykiety (y_pred) dla Gradient Boosting Classifier
# Załóżmy, że y_true i y_pred są zdefiniowane

# Obliczenie precyzji
precision = precision_score(y_test, y_pred, average='macro')

print("Precyzja dla Gradient Boosting Classifier:", precision)



from sklearn.metrics import cohen_kappa_score

# Prawdziwe etykiety (y_true) i przewidywane etykiety (y_pred) dla Gradient Boosting Classifier
# Załóżmy, że y_true i y_pred są zdefiniowane

# Obliczenie współczynnika Kappa Cohena
kappa = cohen_kappa_score(y_test, y_pred)

print("Współczynnik Kappa Cohena dla Gradient Boosting Classifier:", kappa)


######################################################### roc ###############################

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

y_prob = best_gb_clf.predict_proba(X_test)

# 7. Obliczenie krzywej ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(best_gb_clf.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 8. Wyświetlenie krzywej ROC
plt.figure()
for i in range(len(best_gb_clf.classes_)):
    plt.plot(fpr[i], tpr[i], lw=2, label='Klasa {0} (obszar pod krzywą = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Odsetek fałszywie pozytywnych')
plt.ylabel('Odsetek prawdziwie pozytywnych')
plt.title('Krzywa ROC dla klasyfikacji czteroklasowej')
plt.legend(loc="lower right")
plt.show()