import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Load models and scalers
scaler = joblib.load('scaler_ensemble.pkl')
loaded_model = pickle.load(open('sports_prediction_ensemble_model.pkl', 'rb'))

# Load the training features
with open('training_features.pkl', 'rb') as f:
    training_features = pickle.load(f)

def main():
    st.title('Football Player Rating Prediction')

    # Create sliders for all the features used in training
    inputs = {}
    for feature in training_features:
        inputs[feature] = st.slider(feature, 0, 100, 50)

    # Predict function
    def predict_rating_and_confidence(inputs):
        input_data = pd.DataFrame([inputs])
        # Ensure input columns match the training features
        input_data = input_data[training_features]
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        # Make prediction using the loaded model
        predictions = np.array([est.predict(input_scaled)[0] for est in loaded_model.estimators_])
        prediction = np.mean(predictions)
        confidence = np.std(predictions)
        return int(prediction), confidence

    if st.button('Predict'):
        prediction, confidence = predict_rating_and_confidence(inputs)
        st.write(f'Predicted Overall Rating: {prediction}')
        st.write(f'Confidence Score: {confidence:.2f}')

if __name__ == '__main__':
    main()
