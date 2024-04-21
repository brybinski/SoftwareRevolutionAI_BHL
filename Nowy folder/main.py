from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

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

# Definicja endpointu API
@app.post("/convert_to_numpy")
async def convert_to_numpy(input_data: InputData):
    try:
        # Konwersja danych wejściowych na tablicę NumPy
        data_array = np.array(list(input_data.dict().values()))

        # Zwrócenie tablicy NumPy jako odpowiedzi
        return 5
    except Exception as e:
        # Obsługa błędu, jeśli wystąpi
        raise HTTPException(status_code=500, detail=str(e))