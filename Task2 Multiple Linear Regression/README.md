# Multiple Linear Regression on California Housing Prices

## Project Overview

This project implements a Multiple Linear Regression model to predict housing prices in California based on various features. The goal is to build a reliable regression model, evaluate its performance, and visualize the results to understand the relationship between actual and predicted house prices.

## Dataset

The dataset used is the **California Housing Dataset**, which includes several features such as average rooms, population, median income, etc., along with the target variable `MedHouseValue` representing median house prices.

The dataset is loaded directly from the `sklearn.datasets` module (`fetch_california_housing`).

## Steps Implemented

1. **Import Libraries:** Imported necessary Python libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn`.

2. **Load Dataset:** Loaded the California housing dataset and converted it into a `pandas.DataFrame`. Added the target column `MedHouseValue` to the DataFrame.

3. **Inspect Data:** Performed exploratory data analysis (EDA) to understand the data structure and check for missing values or anomalies.

4. **Data Preprocessing:**  
   - Detected outliers and handled them using scaling techniques like StandardScaler.  
   - Prepared the data for model training.

5. **Feature and Target Selection:** Selected relevant features as inputs (X) and the median house value as the target (Y).

6. **Train-Test Split:** Split the data into training (80%) and testing (20%) sets.

7. **Train Linear Regression Model:** Trained the model using the training data.

8. **Make Predictions:** Used the trained model to predict house prices on the test set.

9. **Evaluate the Model:** Calculated evaluation metrics such as:  
   - Mean Squared Error (MSE)  
   - Root Mean Squared Error (RMSE)  
   - R-squared Score (R²)  

10. **Visualize Results:**  
    - Actual vs Predicted Prices scatter plot  
    - Residual plot (Residuals vs Predicted values)  
    - Distribution plot of actual vs predicted prices  

11. **Model Improvement:** Implemented a **Random Forest Regressor** to compare performance and improve accuracy.

12. **Save Model and Outputs:**  
    - Saved the trained Linear Regression model (`LinearRegressionModel.pkl`) using `joblib`.  
    - Created and saved a CSV file (`predicted_house_prices.csv`) containing actual vs predicted house prices for further analysis.

## Outputs

- **Model Performance:**  
  - MSE: ~0.255 (lower is better)  
  - RMSE: ~0.505 (lower is better)  
  - R² Score: ~0.805 (closer to 1 indicates better fit)

- **Files:**  
  - `LinearRegressionModel.pkl`: Serialized model file for future use.  
  - `predicted_house_prices.csv`: CSV file containing actual and predicted house prices side-by-side.

- **Visualizations:**  
  - Scatter plot comparing actual and predicted prices to check model accuracy visually.  
  - Residual plot to observe errors in predictions.  
  - Distribution plot to compare the density of actual and predicted values.

## Technologies and Libraries Used

- Python 3.x  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
