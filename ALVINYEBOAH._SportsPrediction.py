#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import streamlit as st


# In[29]:


# Loading the dataset
test_df = pd.read_csv('players.csv', na_values= "")


# In[32]:


# Select relevant features
best_correlated_columns = ['overall', 'potential', 'age', 'height_cm', 'weight_kg', 'pace', 'shooting', 'passing',
                     'dribbling', 'defending', 'physic', 'attacking_crossing', 'attacking_finishing',
                     'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys',
                     'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
                     'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
                     'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina',
                     'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions',
                     'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure',
                     'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
                     'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning',
                     'goalkeeping_reflexes', 'goalkeeping_speed']

# Calculate correlation with 'overall' and sort
correlation_matrix = test_df[best_correlated_columns].corr()
correlation_overall = correlation_matrix['overall'].sort_values(ascending=False)
correlation_overall


# In[47]:


# Select best correlated features (excluding 'overall' itself)
best_correlated_columns = correlation_overall.index[1:50].tolist()  # top 30 features
best_correlated_columns.append('overall')
best_correlated_columns


# In[48]:


test_df = pd.read_csv('players.csv', usecols=best_correlated_columns)


# In[49]:


# Replacing NAN values with 0
test_df.fillna(0, inplace=True)

# combine instances with overall ratings 92 and 93 into a single class '92'
test_df['overall'] = test_df['overall'].apply(lambda x: 92 if x in [92, 93] else x)


# In[50]:


# Combine similar attributes into single features

# Shooting skills
shooting_attributes = ['shooting', 'power_shot_power', 'power_long_shots', 'attacking_volleys', 'attacking_finishing']
test_df['shooting_skills'] = test_df[shooting_attributes].mean(axis=1)
test_df.drop(columns=shooting_attributes, inplace=True)


# In[51]:


# Mentality attributes
mentality_attributes = ['mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
                        'mentality_vision', 'mentality_penalties', 'mentality_composure']
imputer = SimpleImputer(strategy='mean')
test_df[mentality_attributes] = imputer.fit_transform(test_df[mentality_attributes])
test_df['mentality'] = test_df[mentality_attributes].mean(axis=1)
test_df.drop(columns=mentality_attributes, inplace=True)


# In[52]:


# Technical skills
skill_attributes = ['skill_long_passing', 'skill_ball_control', 'skill_curve', 'skill_fk_accuracy', 'skill_dribbling']
test_df['technical_skills'] = test_df[skill_attributes].mean(axis=1)
test_df.drop(columns=skill_attributes, inplace=True)


# In[53]:


# Goalkeeping abilities
goalkeeping_attributes = ['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                          'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed']
test_df[goalkeeping_attributes] = imputer.fit_transform(test_df[goalkeeping_attributes])
test_df['goalkeeping_ability'] = test_df[goalkeeping_attributes].mean(axis=1)
test_df.drop(columns=goalkeeping_attributes, inplace=True)


# In[54]:


# Transform all features to integer type
test_df = test_df.astype(int)

# Extract target and features
y = test_df['overall']
X = test_df.drop(columns=['overall'])


# In[55]:


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)



# In[57]:


# Train and evaluate models

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f'Mean Absolute Error for Random Forest: {mae_rf}')


# In[63]:


# XGBoost Regressor
from xgboost import XGBRegressor
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f'Mean Absolute Error for XGBoost: {mae_xgb}')


# In[ ]:


# Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
print(f'Mean Absolute Error for Gradient Boosting: {mae_gb}')


# In[65]:


from sklearn.model_selection import GridSearchCV
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_rf = GridSearchCV(estimator=RandomForestRegressor(), param_grid=rf_params, scoring='neg_mean_absolute_error', cv=5)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

# Hyperparameter tuning for XGBoost
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001]
}
grid_xgb = GridSearchCV(estimator=XGBRegressor(), param_grid=xgb_params, scoring='neg_mean_absolute_error', cv=5)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_


# In[66]:


from sklearn.ensemble import VotingRegressor
ensemble_model = VotingRegressor(estimators=[('rf', best_rf), ('xgb', best_xgb)])
ensemble_model.fit(X_train, y_train)
ensemble_predictions = ensemble_model.predict(X_test)
mae_ensemble = mean_absolute_error(y_test, ensemble_predictions)
print(f'Mean Absolute Error for Ensemble Model: {mae_ensemble}')


# In[67]:


# Save the scaler
joblib.dump(scaler, 'scaler_ensemble.pkl')

# Save the ensemble model
import pickle
filename = 'sports_prediction_ensemble_model.pkl'
pickle.dump(ensemble_model, open(filename, 'wb'))

# Load the saved model (for verification)
loaded_model = pickle.load(open(filename, 'rb'))


# In[68]:


training_features = X.columns.tolist()
with open('training_features.pkl', 'wb') as f:
    pickle.dump(training_features, f)

