"""
Wine Price Prediction
Python | Scikit-learn | Pandas

Dataset: Kaggle "Wine Rating and Price" (budnyak/wine-rating-and-price)
The script downloads it automatically with kagglehub.
If the download fails, put Red.csv, White.csv, Rose.csv, Sparkling.csv
in the SAME folder as this script and run again.
"""

import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
TOP_N = 25  # keep the top N categories per categorical column
HERE = os.path.dirname(os.path.abspath(__file__))

# 1. LOAD DATA
def find_data_folder():
    """Try kagglehub first, then fall back to the script's own folder."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("budnyak/wine-rating-and-price")
        print("Dataset downloaded to:", path)
        return path
    except Exception as e:
        print(f"kagglehub download not available ({e}). Looking in script folder instead.")
        return HERE


def load_wine_files(folder):
    frames = []
    for f in glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True):
        part = pd.read_csv(f)
        part.columns = [c.strip().lower() for c in part.columns]
        # only keep files that look like wine data (have a price and a rating)
        if not {"price", "rating"}.issubset(part.columns):
            continue
        base = os.path.basename(f).lower()
        part["wine_type"] = next(
            (t for t in ["red", "white", "rose", "sparkling"] if t in base), "other"
        )
        print(f"Loaded {os.path.basename(f)}: {part.shape}")
        frames.append(part)
    if not frames:
        sys.exit("No wine CSV files found. Put Red.csv, White.csv etc. next to this script.")
    return pd.concat(frames, ignore_index=True)


df = load_wine_files(find_data_folder())
print("\nColumns found:", list(df.columns))
print("Raw shape:", df.shape)

# 2. DATA PREPROCESSING
print("\nMissing values per column:\n", df.isnull().sum())

df = df.drop_duplicates()
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["price", "rating"])      # target and key feature must exist
df = df[df["price"] > 0]

# Remove extreme price outliers (top 1%) so they don't dominate the error metrics
price_cap = df["price"].quantile(0.99)
df = df[df["price"] <= price_cap]

for col in ["country", "region", "winery", "name"]:
    if col not in df.columns:
        df[col] = "Unknown"
    df[col] = df[col].fillna("Unknown").astype(str)

if "numberofratings" not in df.columns:
    df["numberofratings"] = 0
df["numberofratings"] = pd.to_numeric(df["numberofratings"], errors="coerce").fillna(0)

print("Shape after cleaning:", df.shape)

# 3. FEATURE ENGINEERING
# Year can be a number or "N.V." (non-vintage)
year_raw = df["year"] if "year" in df.columns else pd.Series(np.nan, index=df.index)
df["is_non_vintage"] = year_raw.astype(str).str.upper().str.contains("N.V").astype(int)
df["year_num"] = pd.to_numeric(year_raw, errors="coerce")
df["year_num"] = df["year_num"].fillna(df["year_num"].median())
df["wine_age"] = datetime.now().year - df["year_num"]

df["log_num_ratings"] = np.log1p(df["numberofratings"])
df["name_length"] = df["name"].str.len()

# Group rare categories into "Other" to avoid thousands of one-hot columns
for col in ["country", "region", "winery"]:
    top = df[col].value_counts().nlargest(TOP_N).index
    df[col] = df[col].where(df[col].isin(top), "Other")

numeric_features = ["rating", "log_num_ratings", "wine_age", "is_non_vintage", "name_length"]
categorical_features = ["country", "region", "winery", "wine_type"]

X = df[numeric_features + categorical_features]
y = df["price"]

# 4. EXPLORATORY DATA ANALYSIS (plots saved as PNG files)
print("\nPrice summary:\n", df["price"].describe())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(df["price"], bins=50, color="#7b2d43")
axes[0, 0].set_title("Price distribution")
axes[0, 0].set_xlabel("Price ($)")

axes[0, 1].scatter(df["rating"], df["price"], alpha=0.15, color="#7b2d43")
axes[0, 1].set_title("Rating vs Price")
axes[0, 1].set_xlabel("Rating")
axes[0, 1].set_ylabel("Price ($)")

df.groupby("country")["price"].mean().sort_values(ascending=False).head(10).plot(
    kind="bar", ax=axes[1, 0], color="#7b2d43")
axes[1, 0].set_title("Average price by country (top 10)")

corr = df[numeric_features + ["price"]].corr()
im = axes[1, 1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(corr)))
axes[1, 1].set_yticks(range(len(corr)))
axes[1, 1].set_xticklabels(corr.columns, rotation=45, ha="right")
axes[1, 1].set_yticklabels(corr.columns)
axes[1, 1].set_title("Correlation heatmap")
fig.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig(os.path.join(HERE, "eda_plots.png"), dpi=120)
plt.close()
print("Saved EDA plots to eda_plots.png")

# 5. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# 6. TRAIN AND EVALUATE MODELS
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE
    ),
}

results = []
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    predictions[name] = preds

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    results.append({"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse})

# 7. COMPARE MODELS
results_df = pd.DataFrame(results).set_index("Model").round(2)
print("\n===== MODEL COMPARISON =====")
print(results_df)

best_model = results_df["RMSE"].idxmin()
print(f"\nBest model (lowest RMSE): {best_model}")

results_df.to_csv(os.path.join(HERE, "model_results.csv"))

results_df.plot(kind="bar", figsize=(9, 6), rot=0)
plt.title("Model comparison (lower is better)")
plt.ylabel("Error")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "model_comparison.png"), dpi=120)
plt.close()

plt.figure(figsize=(7, 7))
plt.scatter(y_test, predictions[best_model], alpha=0.3, color="#7b2d43")
lims = [0, y_test.max()]
plt.plot(lims, lims, "k--")
plt.xlabel("Actual price ($)")
plt.ylabel("Predicted price ($)")
plt.title(f"Actual vs Predicted - {best_model}")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "actual_vs_predicted.png"), dpi=120)
plt.close()

print("Saved model_results.csv, model_comparison.png, actual_vs_predicted.png")
