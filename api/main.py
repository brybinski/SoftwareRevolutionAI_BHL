from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

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



import dalex as dx
import numpy as np
import warnings


def calculate_shap(model, X, y, instance, cls_num, N=1000):
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    result = []
    for i in range(0, cls_num):
        
        pf = lambda m, d: m.predict_proba(d)[:, i]
        exp = dx.Explainer(model, X, y,  predict_function=pf)
        result.append(exp.predict_parts(instance, type="shap").result)
    warnings.filterwarnings("default", category=UserWarning)
    warnings.filterwarnings("default", category=FutureWarning)

    return result


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
        

        # Zwrócenie tablicy NumPy jako odpowiedzi
        return predicted_class
    except Exception as e:
        # Obsługa błędu, jeśli wystąpi
        raise HTTPException(status_code=500, detail=str(e))