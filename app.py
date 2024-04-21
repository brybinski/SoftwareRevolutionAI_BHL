import pickle
import numpy as np
# Wczytanie modelu z pliku .pkl
with open('model_rf.pkl', 'rb') as f:
    model = pickle.load(f)

predicted_class = model.predict([np.array([0.        , 0.05633803, 0.        , 0.08333333, 1.        ,
       0.5       , 0.        , 0.        , 0.        , 0.        ,
       0.        , 1.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 1.        , 0.        ,
       0.        , 0.        ,0])])[0]


import pandas as pd

data = pd.read_csv("test.csv")
# Przygotowanie pojedynczego rekordu

from sklearn.preprocessing import LabelEncoder

# Label Encoding dla kolumn kategorycznych
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Ever_Married', 'Graduated', 'Spending_Score', 'Segmentation']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])
