# Student-Performance-Prediction-Model

**Project Summary**
The project aims to develop a model that predicts students' grades based on several factors and utilizing a number of machine learning techniques such as data preprocessing, model training and hyperparameter tuning.  

**Data Set Overview** 
Imported as Student Performance Dataset from UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/320/student+performance

studytime: Students' Weekly Study Time (numeric)<br/>
failures: Historical Occurences of Failures (numeric)<br/>
age: Students' Age (numeric)<br/>
Medu: Student's Mother's Education Level (numeric)<br/>
Fedu: Student's Father's Education Level (numeric)<br/>
famrel: Quality of Family Relationship (numeric)<br/>
goout: Outings with Friends (numeric)<br/>
Dalc: Alcholic Consumption on Weekdays (numeric)<br/>
Walc: Alcoholic Consumption on Weekends (numeric)<br/>
health: Current Health Status (numeric)<br/>
absences: Historical Absence Record (numeric)<br/>
G1, G2: First and Second Period Grades (numeric)<br/>
G3: Final Grade (target variable)<br/>

The Data Set also includes a number of other features that are accounted for during the prediction model but listed above are the key features.<br/>
The next section will highlight some key features of the data handling and processing within the program. 

**Phase 1: Loading Data**

**Reading the Data:** Utilizing pd.read_csv, the file path can be read, alongside with sep=';' which is used to allow pandas to recognize ';' as the delimeter instead of ','. 

**Phase 2: Data Preprocessing**

**Missing Data Handling:** df.isnull().sum() is utilized to check for missing data within the data set. This data set has no missing data.<br/> 
**Data Conversion:** pd.get_dummies allows for categorical data (e.g. education level) to be converted to separate binary column to be processed by the model.<br/>
**Feature Scaling:** Using StandardScaler from the scikit library scales all numerical features so they have a mean of 0 and a standard deviation of 1, making the model training more stable and improving performance.

**Phase 3: Data Splitting**

**Train-Test Splitting:** This allows the model to learn from one portion of the data and be evaluated on another, helping to assess its generalization ability.


