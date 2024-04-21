from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Inicjalizacja aplikacji FastAPI
app = FastAPI()

# Definicja modelu danych wejściowych
class InputData(BaseModel):
    Gender: int
    Ever_Married: int
    Age: int
    Graduated: int
    Work_Experience: int
    Spending_Score: int
    Family_Size: int
    Profession_Artist: int
    Profession_Doctor: int
    Profession_Engineer: int
    Profession_Entertainment: int
    Profession_Executive: int
    Profession_Healthcare: int
    Profession_Homemaker: int
    Profession_Lawyer: int
    Profession_Marketing: int
    Var_1_Cat_1: int
    Var_1_Cat_2: int
    Var_1_Cat_3: int
    Var_1_Cat_4: int
    Var_1_Cat_5: int
    Var_1_Cat_6: int
    Var_1_Cat_7: int



## Przygotowanie zbioru treningowego w potrzebnego do algorytmów wyjaśnialności
# Wczytanie danych
data = pd.read_csv('train.csv')

# Usunięcie brakujących wartości
data[['Ever_Married', 'Graduated']] = data[['Ever_Married', 'Graduated']].fillna(value='No')


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

#One-Hot Encoding dla kolumny 'Profession'
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

data


categorical_columns = ['Gender', 'Ever_Married', 'Graduated', 'Spending_Score']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

data



columns_to_remove_outliers = ['Age', 'Work_Experience', 'Family_Size']

# Usunięcie wartości odstających z wybranych kolumn
for col in columns_to_remove_outliers:
    mean = np.mean(data[col])
    std_dev = np.std(data[col])
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

data

# Podział danych na zbiór treningowy i testowy po usunięciu wartości odstających
X = data.drop(['Segmentation'], axis=1)
X = X.drop('ID', axis=1)
y = data['Segmentation']


import dalex as dx
import numpy as np
import warnings


def calculate_shap(model, X, y, instance, cls_num, N=1000):
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


    pf = lambda m, d: m.predict_proba(d)[:, cls_num]
    exp = dx.Explainer(model, X, y,  predict_function=pf)
    result = exp.predict_parts(instance, type="shap").result
    warnings.filterwarnings("default", category=UserWarning)
    warnings.filterwarnings("default", category=FutureWarning)
    key_value_dict = {}

    # Iteracja po wierszach DataFrame i dodawanie każdej pary klucz-wartość do słownika
    for index, row in result.iterrows():
        key_value_dict[row['variable']] = row['contribution']
    return key_value_dict


# Definicja endpointu API
@app.post("/convert_to_numpy")
async def convert_to_numpy(input_data: InputData):
    try:
        # Konwersja danych wejściowych na tablicę NumPy
        data_dict = input_data.dict()
        data_array = np.array(list(data_dict.values()))
        
        


        with open('model_rf.pkl', 'rb') as f:
            model = pickle.load(f)
        predicted_class = model.predict(data_array.reshape(1, -1)).tolist()

        x = str(calculate_shap(model,X, y, data_array, predicted_class[0], 10))
        # Zwrócenie tablicy NumPy jako odpowiedzi
        alph = ['a', 'b', 'c', 'd']
        return ({"segmentation" : alph[predicted_class[0]]}, x)
    except Exception as e:
        # Obsługa błędu, jeśli wystąpi
        raise HTTPException(status_code=500, detail=str(e))