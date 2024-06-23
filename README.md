# Football Player Rating Prediction

This project predicts the overall rating of a football player based on various attributes using an ensemble machine learning model. The application is built using Streamlit for the web interface, and the model is a combination of Random Forest, XGBoost, and Gradient Boosting Regressors.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Features
- Predicts football player ratings based on input attributes
- Provides a confidence score for the predictions
- User-friendly interface with sliders for input

## Requirements
- Python 3.7+
- Streamlit
- scikit-learn
- joblib
- xgboost
- pandas
- numpy
- pickle

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com//alvinyeboah/Intro-to-AI--Lab.git
    cd football-player-rating-prediction
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Ensure you have the model files (`scaler_ensemble.pkl`, `sports_prediction_ensemble_model.pkl`, and `training_features.pkl`) in the project directory.

2. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

3. Open your web browser and go to `http://localhost:8501` to interact with the application.

## Model Training
To train the model, use the provided script (`train_model.py`). This script will load the dataset, preprocess the data, train the models, and save the necessary files for the Streamlit application.

1. Ensure you have the dataset (`players.csv`) in the project directory.

2. Run the training script:
    ```bash
    python train_model.py
    ```

3. The script will generate the model files and save them in the project directory.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
