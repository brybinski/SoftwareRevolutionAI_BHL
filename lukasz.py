
import pickle
import pandas as pd

X_test = pd.read_csv('./test.csv')
X_test = X_test.drop(columns=['ID'])
with open('./model_rf.pkl', 'rb') as f:
    model = pickle.load(f)
    predictions = model.predict(X_test)
    print(predictions)