import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.linear_model import Ridge  
from sklearn.model_selection import GridSearchCV  

#step 1: loading data phase

#read and load dataset into a data frame 
df = pd.read_csv("C:/Users/User/Desktop/Machine Learning Projects/Predicting Student Performance (Regression Project)/student/student-mat.csv", sep=';') 

#print first 5 rows of data
print (df.head())


#step 2: preprocessing data phase

#checking for missing data
print(df.isnull().sum())

#converting variables into dummy variables
df = pd.get_dummies(df, drop_first=True)

#display new structure
print(df.head())

#scaling dataset
#initiating scaler 
scaler = StandardScaler()

#applying scaling to numerical factors 
numerical_features = df.drop('G3', axis=1)  
scaled_features = scaler.fit_transform(numerical_features)

#put it back into data frame
df_scaled = pd.DataFrame(scaled_features, columns=numerical_features.columns)

#target variable = G3, put it back into dataframe 
df_scaled['G3'] = df['G3']

print(df_scaled.head())

#step 3: data splitting phase

#split features as X and target as Y
X = df_scaled.drop('G3', axis=1)  # features (input)
Y = df_scaled['G3']  # target (output)

#testing size is 20% 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)

#step 4: model training phase 

#model initialization 
model = LinearRegression()

#training model for training data
model.fit(X_train, Y_train)

#step 5: model evaluation

#predict based on test data
Y_pred = model.predict(X_test)

#calculating mse 
mse = np.mean((Y_test - Y_pred)**2)
print(f"Mean Squared Error: {mse}")


#step 6: tuning and experimenting

#setting a ridge regression model
ridge = Ridge()

#defining hyperparameter grid 
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}

#setting up grid search
grid_search = GridSearchCV(ridge, param_grid, cv=5)

#model training to find best Alpha value
grid_search.fit(X_train, Y_train)

#print best parameters 
print(grid_search.best_params_)