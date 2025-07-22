# Titanic Survival Prediction Using Decision Tree Classifier

This project uses a Decision Tree Classifier to predict whether a passenger survived the Titanic disaster based on features like age, gender, ticket class, and fare. The dataset used is a cleaned version of the Titanic dataset.

## Task Overview

## Goal: Predict the Survived status (0 = Not Survived, 1 = Survived)

## Algorithm Used: Decision Tree Classifier (from sklearn.tree)


## Steps Followed

### 1. Load the Cleaned Dataset

Used a preprocessed Titanic dataset from Task 1

Included scaled columns like age_scaled and fare_scaled



### 2. Separate Features and Target

X = All columns except Survived

y = Survived column (target variable)



### 3. Train-Test Split

Used train_test_split() with 80% training and 20% testing data



### 4. Initialize and Train the Decision Tree Model

Used DecisionTreeClassifier(random_state=42)

Trained using model.fit(X_train, y_train)



### 5. Make Predictions

Predicted outcomes on X_test and stored them in y_pred



### 6. Evaluate the Model

Evaluated using:

Accuracy Score

Confusion Matrix

Classification Report




### 7. Visualize the Decision Tree

Used plot_tree() from sklearn with Matplotlib

Showed how splits were made based on features



### 8. Save the Model

Used joblib to save the trained model as decision_tree_model.pkl




### Output Files

decision_tree_model.pkl – Trained Decision Tree model




### Key Learnings

Decision Trees are interpretable and handle both numerical and categorical data well

No need for scaling or encoding if the model can handle original inputs

Confusion matrix and classification reports are useful for deeper evaluation beyond accuracy


### Technologies Used

Python

Scikit-learn

Pandas, NumPy

Matplotlib

Google Colab
