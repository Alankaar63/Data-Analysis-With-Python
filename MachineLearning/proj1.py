import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib
matplotlib.use('Agg') 


data_cleaned = pd.read_csv('signal_metrics.csv')
print(data_cleaned.columns)

data_cleaned['Failure'] = (
    (data_cleaned['Signal Strength (dBm)'] < -93) |
    (
        (data_cleaned['BB60C Measurement (dBm)'] < -93) &
        (data_cleaned['srsRAN Measurement (dBm)'] < -93) &
        (data_cleaned['BladeRFxA9 Measurement (dBm)'] < -93) &
        (data_cleaned['Latency (ms)'] > 90)
    )
).astype(int)

for col in data_cleaned.select_dtypes(include=['object']).columns:
    try:
        data_cleaned[col] = pd.to_datetime(data_cleaned[col])
        data_cleaned[col] = data_cleaned[col].astype(int) // 10**9 
    except:
        pass  


data_cleaned['Network Type'] = data_cleaned['Network Type'].fillna('Unknown')


ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_nt = ohe.fit_transform(data_cleaned[['Network Type']])
encoded_nt_df = pd.DataFrame(encoded_nt, columns=ohe.get_feature_names_out(['Network Type']))
data_cleaned = pd.concat([data_cleaned, encoded_nt_df], axis=1)
data_cleaned.drop(columns=['Network Type'], inplace=True)


drop_cols = ['Sr.No.', 'Locality']
data_cleaned = data_cleaned.drop(columns=[col for col in drop_cols if col in data_cleaned.columns])


X = data_cleaned.drop(columns=['Failure'])
y = data_cleaned['Failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

import joblib

joblib.dump(model,"model.pkl")
print("saved successfully!")




