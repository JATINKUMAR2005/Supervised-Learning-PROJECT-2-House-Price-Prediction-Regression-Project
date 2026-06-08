# House Price Prediction - Regression Project
 
**Name:** Jatin  
**Subject:** Machine Learning  
**Institution:** Red & White Skill Education  
**Topic:** Supervised Learning - Regression with Regularization and Cross Validation
 
---
 
## About the Project
 
In this project I built a machine learning model to predict house prices based on features like area, number of bedrooms, location score, and crime rate. I used different regression techniques and compared which model works best.
 
The main goal was to understand how regularization (Ridge and Lasso) works and why cross validation is better than a single train-test split.
 
---
 
## Dataset
 
- **File:** `Advanced_Regression_HousePrice_Dataset_3800_csv.csv`
- **Rows:** 3800
- **Columns:** 12
- **Target Variable:** `house_price_inr`
### Features used for prediction:
 
| Column | Description |
|---|---|
| area_sqft | Size of the property in square feet |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| location_score | Score of the location (higher = better) |
| property_age | Age of the property in years |
| distance_city_km | Distance from city center in km |
| near_school | 1 if near school, 0 if not |
| near_metro | 1 if near metro, 0 if not |
| crime_rate_index | Crime rate in that area (lower = safer) |
 
Columns `property_id` and `sale_date` were dropped as they are not useful for prediction.
 
---
 
## Project Structure
 
```
house-price-prediction/
│
├── Advanced_Regression_HousePrice_Dataset_3800_csv.csv   # dataset
├── student_regression.ipynb                               # main notebook
└── README.md                                              # this file
```
 
---
 
## What I Did (Step by Step)
 
1. **Loaded the dataset** and checked its shape, data types, and missing values
2. **EDA** - plotted distributions of all features and a correlation heatmap
3. **Train-Test Split** - 80% training, 20% testing
4. **Feature Scaling** - used StandardScaler (needed for Ridge and Lasso)
5. **Ridge Regression** - tried multiple alpha values, picked the best one
6. **Lasso Regression** - tried multiple alpha values, checked which features became zero
7. **K-Fold Cross Validation** - used 5 folds to check model consistency
8. **Decision Tree** - trained without scaling, checked depth
9. **Random Forest** - trained with 100 trees, checked feature importance
10. **Final Comparison** - compared all 4 models using R2, RMSE, MAE
---
 
## Models Used
 
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)
- Decision Tree Regressor
- Random Forest Regressor
---
 
## Results
 
| Model | Test R2 | Notes |
|---|---|---|
| Ridge Regression | ~0.87 | Good baseline, all features kept |
| Lasso Regression | ~0.87 | Some features set to zero |
| Decision Tree | ~0.85 | Slightly lower, single tree |
| Random Forest | ~0.93 | Best model, handles non-linear patterns |
 
> Note: Exact scores may vary slightly due to randomness in the data
 
**Best Model: Random Forest** — highest R2 and lowest RMSE
 
---
 
## Key Observations
 
- `area_sqft` and `location_score` are the most important features for predicting price
- `crime_rate_index` has a negative effect on price — higher crime = lower price
- Lasso removed some less important features automatically (coefficient = 0)
- Ridge kept all features but shrunk their weights
- Random Forest gave the best results because it handles non-linear relationships
---
 
## Libraries Used
 
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```
 
Install using:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```
 
---
 
## How to Run
 
1. Download or clone this repository
2. Make sure the dataset CSV file is in the same folder as the notebook
3. Open `student_regression.ipynb` in Jupyter Notebook or JupyterLab
4. Run all cells from top to bottom
---
 
## Theory Covered
 
- What is Regularization and why it is needed
- Difference between Ridge (L2) and Lasso (L1)
- What is Cross Validation and its types (K-Fold, Stratified, LOO, TimeSeriesSplit)
- Why tree-based models do not need feature scaling
