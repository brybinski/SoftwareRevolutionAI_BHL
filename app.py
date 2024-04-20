import pickle

# Wczytanie modelu z pliku .pkl
with open('model_rf.pkl', 'rb') as f:
    model = pickle.load(f)