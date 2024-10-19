import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.linear_model import Ridge  
from sklearn.model_selection import GridSearchCV  

df = pd.read_csv("student-mat.csv", sep=';') 
print (df.head())


print(df.isnull().sum())
df = pd.get_dummies(df, drop_first=True)
print(df.head())


scaler = StandardScaler()
numerical_features = df.drop('G3', axis=1)  
scaled_features = scaler.fit_transform(numerical_features)


df_scaled = pd.DataFrame(scaled_features, columns=numerical_features.columns)
df_scaled['G3'] = df['G3']

print(df_scaled.head())

X = df_scaled.drop('G3', axis=1)  # features (input)
Y = df_scaled['G3']  # target (output)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

mse = np.mean((Y_test - Y_pred)**2)
print(f"Mean Squared Error: {mse}")

ridge = Ridge()
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(ridge, param_grid, cv=5)

grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)

#understanding data correlation

correlation_matrix = df_scaled.corr()
high_corr = correlation_matrix[(correlation_matrix > 0.5) | (correlation_matrix < -0.5)]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.heatmap(correlation_matrix, fmt=".2f", annot=True, cmap='coolwarm', xticklabels=1, 
            yticklabels=1, annot_kws={"size": 8}, cbar_kws={"label": "Correlation"}, ax=axes[0])
axes[0].set_title("Correlation Heatmap of Student Dataset", fontsize=16)
axes[0].set_xlabel("Features", fontsize=12)
axes[0].set_ylabel("Features", fontsize=12)
axes[0].tick_params(axis='x', rotation=45, labelsize=8)
axes[0].tick_params(axis='y', rotation=0, labelsize=8)


sns.heatmap(high_corr, annot=True, cmap='coolwarm', ax=axes[1])
axes[1].set_title("High Correlation Heatmap", fontsize=16)
axes[1].tick_params(axis='x', labelsize=8)
axes[1].tick_params(axis='y', labelsize=8)


plt.tight_layout()
plt.show()
