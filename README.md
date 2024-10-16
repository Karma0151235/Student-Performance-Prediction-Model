# Student-Performance-Prediction-Model #

This project aims to predict students' final grades using a variety of demographic and academic data (such as study time, prior grades, family background, and more). The problem is treated as a regression task, where we predict a continuous value (the final grade).

## Project Overview ##
By leveraging machine learning techniques, this project uses Python to build predictive models that estimate students' final grades based on input data. We employ a linear regression model as well as a tuned Ridge regression model to enhance performance.

## To Run the Model ##

**1. Clone the Repository:**<br/>
Start by cloning this repository to your local machine.
`git clone https://github.com/your-repo/student-performance-prediction.git`
`cd student-performance-prediction`

**2. Install Required Libraries:**<br/>
Install the required Python libraries using `pip` or `conda`:
`pip install pandas numpy scikit-learn`

**3. Loading the Dataset:**<br/>
The dataset `student-mat.csv` should be placed in the appropriate folder. Ensure the file is located in the correct path as indicated in the script, or modify the file path in `main.py`:
`df = pd.read_csv("data/student-mat.csv", sep=';')`

**4. Run the Script:**<br/>
Run the script on your preferred IDE or run it manually with the command below in the terminal. 
`python main.py`

These steps allow you to set up the required libraries and data set for you to run the Python script to load the data, preprocess it, train the models, and evaluate the performance. 




