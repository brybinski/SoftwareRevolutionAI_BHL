from fastapi import FastAPI

# Tworzenie instancji aplikacji FastAPI
app = FastAPI()

# Definicja endpointu API
@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}