import requests
import pandas as pd
import numpy as np


host = '0.0.0.0'
port = '8001'

url = f'http://{host}:{port}/invocations'

headers = {
    'Content-Type': 'application/json',
}

array = np.array([-0.001882, -0.044642, -0.051474,
                  -0.026328, -0.008449, -0.019163,
                  0.074412, -0.039493, -0.068330,
                  -0.092204])

print(array.shape)

# test_data is a Pandas dataframe with data for testing the ML model
test_data = pd.DataFrame([array], columns=['age', 'sex', 'bmi',
                                           'bp', 's1', 's2',
                                           's3', 's4', 's5',
                                           's6'])

print(test_data.head())
http_data = test_data.to_json(orient='split')

r = requests.post(url=url, headers=headers, data=http_data)

print(f'Predictions: {r.text}')
