# Wine Price Prediction

Predicts wine prices using Python, Scikit-learn and Pandas. The project cleans a real-world dataset, engineers features, runs exploratory analysis, and compares three regression models using MAE, MSE and RMSE to pick the most accurate one.

## Dataset

[Wine Rating and Price](https://www.kaggle.com/datasets/budnyak/wine-rating-and-price) from Kaggle (Vivino data). It has separate files for Red, White, Rose and Sparkling wines with name, country, region, winery, rating, number of ratings, price and year.

## What the script does

1. **Load data:** downloads the dataset with `kagglehub` and combines all wine files.
2. **Preprocess:** removes duplicates and missing values, and drops the top 1% of prices as outliers.
3. **Feature engineering:** wine age, non-vintage flag, log of number of ratings, name length, and wine type. Rare countries, regions and wineries are grouped into "Other".
4. **EDA:** price distribution, rating vs price, average price by country, correlation heatmap.
5. **Models:** Linear Regression, Decision Tree, Random Forest.
6. **Evaluation:** compares models with MAE, MSE and RMSE and selects the one with the lowest RMSE.

## Setup and run (Windows)
How to run

Put the dataset CSV files and the code in the same folder. Dataset: Wine Rating and Price (Kaggle). Your folder should look like this:
   wine-project/
      wine_price_prediction.py
      Red.csv
      White.csv
      Rose.csv
      Sparkling.csv
      
Open a terminal in that folder and install the libraries (first time only):
   pip install kagglehub pandas scikit-learn matplotlib numpy
Run the code:
   python wine_price_prediction.py
Output

The results are saved in the same folder:

eda_plots.png - exploratory plots
model_comparison.png - MAE, MSE and RMSE for each model
actual_vs_predicted.png - predictions from the best model
model_results.csv - metrics table
Tech stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib


