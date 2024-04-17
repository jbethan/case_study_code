#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config

from sklearn.model_selection import GridSearchCV

import tensorflow as tf

import seaborn as sns
import calendar
from datetime import datetime
import numpy as np

#%%
import shap
shap.initjs()

#%%
model_data = pd.read_csv('clean_data.csv')

#%%
model_data['Date'] = pd.to_datetime(model_data['Date'])
model_data['Month'] = model_data['Date'].dt.month
#%%
set_config(transform_output="pandas")
#%%
# Select features
X = model_data[[#'tempmax',
                  #'tempmin',
                  'temp',	
                  #'feelslikemax',	
                  #'feelslikemin',	
                  #'feelslike',	
                  #'dew',	
                  'humidity',	
                  'precip',	
                  #'precipprob', 
                  #'precipcover',	
                  'preciptype',	# categorical
                  'snow',	
                  'snowdepth',	
                  'windgust',	
                  #'windspeed',	
                  #'winddir',	
                  #'sealevelpressure',	
                  'cloudcover',	
                  'visibility',	
                  #'solarradiation',	
                  #'solarenergy',	
                  'uvindex',
                  'moonphase',
                  #'conditions', #categorical
                  #'icon', #categorical
                  'IsFullMoon', #categorical
                  'Holiday', #categorical
                  'Month' # categorical
                  ]]
y = model_data['isViolent']

X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
num_features = [#'tempmax',
                  #'tempmin',
                  'temp',	
                  #'feelslikemax',	
                  #'feelslikemin',	
                  #'feelslike',	
                  #'dew',	
                  'humidity',	
                  'precip',	
                  #'precipprob', 
                  #'precipcover',		
                  'snow',	
                  'snowdepth',	
                  'windgust',	
                  #'windspeed',	
                  #'winddir',	
                  #'sealevelpressure',	
                  'cloudcover',	
                  'visibility',	
                  #'solarradiation',	
                  #'solarenergy',	
                  'uvindex',
                  'moonphase'
                ]

cat_features = ['preciptype',
                  #'conditions',
                  #'icon',
                  'IsFullMoon', 
                  'Holiday',
                  'Month'
                ] 

numerical = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('polynomial', PolynomialFeatures(degree = 2, include_bias=False)),
    ('standardize', StandardScaler()),
    ('percent', SelectPercentile(f_regression, percentile=35))
    ])

categorical = Pipeline(steps=[
    ('impute2', SimpleImputer(strategy='most_frequent')),
    #('one_hot', OneHotEncoder(sparse_output=False))
    ('one_hot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    #('percent', SelectPercentile(f_regression, percentile=40))
    ])

preprocessor = ColumnTransformer(
    transformers=
        (
            ("numeric", numerical, num_features),
            ("categorical", categorical, cat_features)
        ),
        verbose_feature_names_out=False,
)

# Update X_train & X_test
X_train_pipe = preprocessor.fit_transform(X_train, y_train)
X_test_pipe = preprocessor.transform(X_test)

# %%
models = {
    "Linear Regression": LinearRegression(),
    #"Decision Tree": DecisionTreeRegressor(random_state=42),
    #"Random Forest": RandomForestRegressor(random_state=42),
    #"K-Nearest Neighbors": KNeighborsRegressor(),
    #"SVM (RBF Kernel)": SVR(kernel='rbf'),
    "SVM (Linear Kernel)": SVR(kernel='linear'),
    #"SVM (Poly Kernel)": SVR(kernel='poly'),
    #"SVM (Sigmoid Kernel)": SVR(kernel='sigmoid')
}

# Train and evaluate models
for name, model in models.items():
    # Fit model & predict
    model.fit(X_train_pipe, y_train)
    y_pred = model.predict(X_test_pipe)
    
    # Calculate mean squared error (lower = better)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name}: Mean Squared Error = {mse}")

# %%
# Testing SVM Parameters
# Parameters to test
test_params = {
    'C': [7.25, 7.5, 7.75],
    'epsilon': [1.7, 1.75, 1.8]
}

# GridSearchCV
svm_model = SVR(kernel='linear')
grid_search = GridSearchCV(svm_model, test_params, cv=4, scoring='neg_mean_squared_error')
grid_search.fit(X_train_pipe, y_train)

# Best Hyperparameters & Score
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_pipe)
best_score = mean_squared_error(y_test, y_pred)
print("Best Param: Mean Squared Error = ", best_score)

#%%
#SVM Post-Tuning
svm_model = SVR(kernel='linear', C=7.25, epsilon=1.75)
svm_model.fit(X_train_pipe, y_train)
y_pred = svm_model.predict(X_test_pipe)
mse = mean_squared_error(y_test, y_pred)
print(f"SVM (Tuned): Mean Squared Error = ", mse)

#%% 
# Check Cross Validation for Overfitting
cv_scores = cross_val_score(svm_model, X_train_pipe, y_train, cv=4, scoring='neg_mean_squared_error')
average_cv_score = -cv_scores.mean()
print(f"Avg CV Score: Mean Squared Error = ", average_cv_score)

#%%
# SVM Feature Importance
# Coefficient for linear relationship between feature and target variable

# Set Feature & Importance 
feature = svm_model.feature_names_in_
importance = svm_model.coef_.flatten()

# Output results
svm_importance = pd.DataFrame({'Feature': feature, 'Importance': importance})
svm_importance = svm_importance.sort_values(by='Importance', ascending=False)
print("SVM Feature Importance:")
for index, row in svm_importance.iterrows():
    print(f"{row['Feature']}: {row['Importance']}")

#%%
#svm_importance.to_csv('svm_importance.csv', index=False)
    
# %%
holiday_importance = svm_importance[svm_importance['Feature'].str.startswith('Holiday_')]

# Print feature importance for Holidays
print("Feature Importance for Holidays:")
for index, row in holiday_importance.iterrows():
    print(f"{row['Feature']}: {row['Importance']}")

#%%
month_importance = svm_importance[svm_importance['Feature'].str.startswith('Month_')]

# Print feature importance for Holidays
print("Feature Importance for Holidays:")
for index, row in month_importance.iterrows():
    print(f"{row['Feature']}: {row['Importance']}") 
#%%
fullmoon_importance = svm_importance[svm_importance['Feature'].str.contains('moonphase|^IsFullMoon')]

# Print feature importance for Holidays
print("Feature Importance for Moonphase:")
for index, row in fullmoon_importance.iterrows():
    print(f"{row['Feature']}: {row['Importance']}")
#%%
svm_num = pd.read_csv('svm_num_importance.csv')

#%%
svm_num.head()
svm_num.columns
#%%
def waterfall_plot(categories, values):

    cum_values = np.cumsum(values)

    data = pd.DataFrame({'Categories': categories, 'Values': values, 'Cumulative Values': cum_values})

    plt.figure(figsize=(6, 10))
    plt.barh(data['Categories'].astype(str), data['Values'],align='center', alpha=0.7, color='purple', label='Features')

    plt.xlabel('Weather')
    plt.ylabel('Feature Importance')
    plt.title('Distribution of Feature Importance by Weather')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.grid(axis='y')
    
    plt.show()

waterfall_plot(svm_num['Feature'], svm_num['Importance'])

#%%
fig, ax1 = plt.subplots(1, 1, figsize=(6, 10))

ax1.barh(svm_num["Feature"].astype(str), svm_num['Importance'],align='center', alpha=0.7, 
         color=['red','purple', 'red', 'red', 'blue', 'blue', 'red', 'red', 'red', 'blue', 'blue', 'red', 'blue', 'blue', 'purple', 'blue', 'purple', 'blue', 'blue'], 
         label='Features')

ax1.set_xlabel('Feature Importance')
ax1.set_ylabel('Weather')
ax1.set_title('Distribution of Feature Importance by Weather')
ax1.invert_yaxis()
    
plt.show()

# %%
# SHAP for SVM
# Average Marginal Contribution of each Feature

# Add SHAP Explainer
explainer = shap.Explainer(svm_model, X_train_pipe)
shap_values = explainer.shap_values(X_train_pipe)
#shap_importance = np.abs(np.mean(shap_values, axis=0))
shap_importance = np.abs(np.mean(shap_values, axis=0))

# Output results
svm_shap_importance = pd.DataFrame({'Feature': feature, 'Importance': shap_importance})
svm_shap_importance = svm_shap_importance.sort_values(by='Importance', ascending=False)
print("SHAP Feature Importance for SVM")
for index, row in svm_shap_importance.iterrows():
    print(f"{row['Feature']}: {row['Importance']}")

#%%
svm_shap_importance.to_csv('svm_shap_importance.csv', index=False)
#%%
shap_bin = explainer(X_train_pipe)       
shap.plots.waterfall(shap_bin[0])

#%%
shap.plots.bar(shap_bin[0], show_data=True)

# %%
holiday_shap_importance = svm_shap_importance[svm_shap_importance['Feature'].str.startswith('Holiday_')]

# Print feature importance for Holidays
print("Feature Importance for Holidays:")
for index, row in holiday_shap_importance.iterrows():
    print(f"{row['Feature']}: {row['Importance']}")

#%%
    
# %%
# Random Forest Feature Importance
#rf_model = RandomForestRegressor(random_state=42)
#rf_model.fit(X_train_pipe, y_train)

# Set feature & importance
#feature = X.columns
#importance = rf_model.feature_importances_

# Output results
#rf_importance = pd.DataFrame({'Feature': feature, 'Importance': importance})
#rf_importance = rf_importance.sort_values(by='Importance', ascending=False)
#print("RF Feature Importance:")
#for index, row in rf_importance.iterrows():
#    print(f"{row['Feature']}: {row['Importance']}")


# %%
#epochs=10
epochs=35
batch_size=64

# FNN model
tf.random.set_seed(42)   
fnn_model = tf.keras.Sequential()
fnn_model.add(tf.keras.layers.InputLayer(input_shape=X_train_pipe.shape[1:]))
fnn_model.add(tf.keras.layers.Flatten())
fnn_model.add(tf.keras.layers.Dense(300, activation='relu'))
fnn_model.add(tf.keras.layers.Dropout(rate=0.2))
fnn_model.add(tf.keras.layers.Dense(100, activation='relu'))
fnn_model.add(tf.keras.layers.Dropout(rate=0.2))
fnn_model.add(tf.keras.layers.Dense(10))


# Compile & Train
fnn_model.compile(optimizer='adam', loss='mean_squared_error')
fnn_model.fit(X_train_pipe, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Output results
mse = fnn_model.evaluate(X_test_pipe, y_test)
print(f"FNN: Mean Squared Error = {mse}")

# %%
#tf.keras.backend.clear_session()

# %%
