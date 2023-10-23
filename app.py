from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


#Load your time series data into a Pandas DataFrame. Ensure that your data is in the right format with a timestamp index.
data = pd.read_excel('cleaned_dataset.xlsx', parse_dates=True)

# Feature selection
column_random_regressor=['Year','Number of ATMs per 100,000 adults',
'Number of depositors with commercial banks per 1,000 adults',
'Number of borrowers from commercial banks per 1,000 adults',
'Value of mobile money transactions (during the reference year) (% of GDP)']

#Selecting the X and y features
new_subset = data[column_random_regressor]
X = new_subset.drop(columns=['Value of mobile money transactions (during the reference year) (% of GDP)'])
y = new_subset['Value of mobile money transactions (during the reference year) (% of GDP)']

# Model training
model = RandomForestRegressor()
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html', prediction="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form['year']),float(request.form['atms']), float(request.form['depositors']), float(request.form['borrowers'])]
        scaled_data = scaler.transform([data])
        prediction = model.predict(scaled_data)[0]
        return render_template('index.html', prediction=f"Predicted Value: {prediction:.2f}")
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

