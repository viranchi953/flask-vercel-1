from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

app = Flask(__name__)

# Load the trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
# rfc_model = pickle.load(open('rfc_model.pkl', 'rb'))
minmax_scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
stand_scaler = pickle.load(open('standscaler.pkl', 'rb'))
imputer = SimpleImputer(strategy='mean')

# Load the dataset
data = pd.read_csv("crop_data.csv") 

# Perform preprocessing
X = data[['STATE', 'SOIL_TYPE', 'N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL']]
y = data['CROP']

X = pd.get_dummies(X) 
X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Sample data for initializing columns in the web form
sample_data = {'STATE': 'maharashtra', 'SOIL_TYPE': 'black', 'N_SOIL': 93, 'P_SOIL': 72, 'K_SOIL': 69,
               'TEMPERATURE': 33, 'HUMIDITY': 88, 'ph': 8, 'RAINFALL': 80}

@app.route('/', methods=['GET', 'POST'])
def predict_crop():
    if request.method == 'POST':
        user_input = {
            'STATE': request.form['state'].lower(),
            'SOIL_TYPE': request.form['soil_type'].lower(),
            'N_SOIL': float(request.form['n_soil']),
            'P_SOIL': float(request.form['p_soil']),
            'K_SOIL': float(request.form['k_soil']),
            'TEMPERATURE': float(request.form['temperature']),
            'HUMIDITY': float(request.form['humidity']),
            'ph': float(request.form['ph']),
            'RAINFALL': float(request.form['rainfall']),
        }

        user_input_df = pd.DataFrame([user_input])
        user_input_df = pd.get_dummies(user_input_df)

        # Fit and transform the imputer on X_train
        X_train_imputed = imputer.fit_transform(X_train)
        
        # Reindex with columns from X_train
        user_input_df = user_input_df.reindex(columns=X_train.columns, fill_value=0)
        
        # Transform the user input data using the fitted imputer
        user_input_df = imputer.transform(user_input_df)

        predicted_crops_proba = model.predict_proba(user_input_df)
        max_prob = max(predicted_crops_proba[0])
        scaled_probs = [prob / max_prob for prob in predicted_crops_proba[0]]

        crops_with_scaled_proba = [(crop, scaled_prob) for crop, scaled_prob in zip(model.classes_, scaled_probs)]
        crops_with_scaled_proba = sorted(crops_with_scaled_proba, key=lambda x: x[1], reverse=True)

        return render_template('index.html', user_input=user_input, crops_with_scaled_proba=crops_with_scaled_proba)

    return render_template('index.html', user_input=sample_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)