# Student-Performance-Prediction-Model #

This project aims to predict students' final grades using a variety of demographic and academic data (such as study time, prior grades, family background, and more). The problem is treated as a regression task, where we predict a continuous value (the final grade).

## Project Overview ##
By leveraging machine learning techniques, this project uses Python to build predictive models that estimate students' final grades based on input data. We employ a linear regression model as well as a tuned Ridge regression model to enhance performance.

## To Run the Model ##

**1. Clone the Repository:**<br/>
Start by cloning this repository to your local machine.<br/>
`git clone https://github.com/your-repo/student-performance-prediction.git`
`cd student-performance-prediction`

**2. Install Required Libraries:**<br/>
Install the required Python libraries using `pip` or `conda`:<br/>
`pip install pandas numpy scikit-learn`

**3. Loading the Dataset:**<br/>
The dataset `student-mat.csv` should be placed in the appropriate folder. Ensure the file is located in the correct path as indicated in the script, or modify the file path in `main.py`:<br/>
`df = pd.read_csv("student-mat.csv", sep=';')`

**4. Run the Script:**<br/>
Run the script on your preferred IDE or run it manually with the command below in the terminal.<br/>
`python main.py`

These steps allow you to set up the required libraries and data set for you to run the Python script to load the data, preprocess it, train the models, and evaluate the performance.

## Understanding the Steps ##

**Step 1: Data Loading**<br/>
The purpose of this step is to load the data set using pandas, and viewing the first 5 rows of the unaltered data set.<br/>
`df = pd.read_csv("data/student-mat.csv", sep=';')`<br/>
`print(df.head())`

**Step 2: Data Preprocessing**<br/>
* Handle missing data and convert categorical variables into dummy variables using `pd.get_dummies()`.<br/>
* Scale the features to standardize the input using `StandardScaler` from `scikit-learn.`<br/>
* The target variable, **G3** (final grade), is kept separate.<br/>
`df = pd.get_dummies(df, drop_first=True)`<br/>
`scaler = StandardScaler()`<br/>
`numerical_features = df.drop('G3', axis=1)`<br/>
`scaled_features = scaler.fit_transform(numerical_features)`

**Step 3: Data Splitting**<br/>
This phase splits the data into training and test sets, where 20% of the data is used for testing.<br/>
`X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)`

**Step 4: Model Training and Evaluating**<br/>
* Train a Linear Regression model using the training data.<br/>
* Evaluate the model using Mean Squared Error (MSE) as the metric.<br/>
`model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
mse = np.mean((Y_test - Y_pred)**2)
print(f"Mean Squared Error: {mse}")`





