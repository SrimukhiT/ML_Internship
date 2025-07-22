 # Task 1  Data Preprocessing
 # Titanic Data Preprocessing for Machine Learning(with Logistic Regression Prediction)
## Titanic Survival Prediction using Logistic Regression
This project uses the Titanic dataset to build a logistic regression model that predicts passenger survival based on various features. The dataset is taken from Kaggle's Titanic challenge and has been processed through multiple steps for effective training and evaluation.

### Dataset Used:
Source: Kaggle - Titanic: Machine Learning from Disaster

### Files:
train.csv, test.csv

### Project Steps
#### 1. Data Loading

Loaded train.csv and test.csv using pandas.

Created backup copies using .copy() to preserve original data.

#### 2. Exploratory Data Analysis (EDA)

Checked for missing values using .isnull().sum().

Displayed data info using .info() to identify data types.

#### 3. Handling Missing Values

Imputed missing values in Age, Embarked, and Fare using median or most frequent methods.

#### 4.Removing duplicate rows

Checked for duplicate rows in both the datasets using .duplicated().sum()

#### 5.Convert datatypes

Checked the suitable datatypes of each column

#### 6.Encoding Categorical Variables

Used LabelEncoder to convert non-numeric columns like Sex and Embarked into numeric values.

#### 7. Feature Scaling

Applied RobustScaler on numeric columns (Age, Fare) to reduce the impact of outliers.

Same scaler was used later for transforming test data to match scaling.

#### 8.Detected and Handle Outliers

Visualizing outliers in 'Fare' ,'Age' using Seaborn Boxplots sns.boxplot(x=train_df['Fare','Age'])

Applying RobustScaler scaler = RobustScaler() train_df[['Age_scaled', 'Fare_scaled']] = scaler.fit_transform(train_df[['Age', 'Fare']])

#### 9. Feature Selection(Drop Irrelevant Columns)

Dropped irrelevant columns such as PassengerId, Name, Cabin, Ticket.

#### 10. Train-Test Split

Separated features (X) and target (y) by removing the Survived column.

Used train_test_split to divide the data into training (80%) and validation (20%) sets.

#### 11. Model Training

Used LogisticRegression from sklearn to train the model on training data.

#### 12. Model Evaluation

Evaluated model on validation data using:

Accuracy Score

Classification Report: Precision, Recall, F1 Score

#### 13. Model Export & Prediction

Exported the trained model using joblib.dump() to a .pkl file (titanic_model.pkl).

Cleaned and scaled test data to match training format.

Loaded model using joblib.load() and made predictions using .predict() on processed test data.

### Tools & Libraries Used
Python

pandas

numpy

scikit-learn

joblib

matplotlib

seaborn boxplot

### Output
Model predicts survival (1 = Survived, 0 = Not Survived).

Predictions are based on cleaned and processed features using logistic regression(stored in submission.csv).
