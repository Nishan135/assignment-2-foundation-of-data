# BISHAL BASNET - LINEAR REGRESSION 2.2
# Predict Goals Scored by a Team in a Match

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
csv_path = Path(__file__).resolve().parent / "wc2026_team_goals.csv"
df = pd.read_csv(csv_path)

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Define features (X) and target (y)
features = ['fifa_rank', 'avg_goals_scored_prev5', 'avg_goals_conceded_prev5',
            'days_rest', 'stage', 'home_away', 'opponent_fifa_rank', 
            'head_to_head_win_rate']
target = 'goals_scored'

X = df[features]
y = df[target]

print(f"\nFeatures: {features}")
print(f"Target: {target}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Descriptive statistics of target
print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS - GOALS SCORED")
print("=" * 70)
print(y.describe())

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {X_train.shape[0]} rows")
print(f"Test set: {X_test.shape[0]} rows")

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Mean goals in test set: {y_test.mean():.2f}")

# Show coefficients
print("\n" + "=" * 70)
print("REGRESSION COEFFICIENTS")
print("=" * 70)
for feature, coef in zip(features, model.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

# Feature importance visualization
plt.figure(figsize=(10, 6))
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')
plt.title("Feature Importance in Predicting Goals Scored")
plt.axvline(x=0, color='red', linestyle='--')
plt.tight_layout()
plt.savefig("bishal_regression_features.png", dpi=150)
print("\nSaved: bishal_regression_features.png")

# Actual vs Predicted scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Goals Scored")
plt.ylabel("Predicted Goals Scored")
plt.title("Actual vs Predicted Goals Scored")
plt.tight_layout()
plt.savefig("bishal_regression_prediction.png", dpi=150)
print("Saved: bishal_regression_prediction.png")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"The model explains {r2*100:.2f}% of the variance in goals scored.")
if r2 > 0.5:
    print("This is a reasonable model for predicting goals.")
elif r2 > 0.3:
    print("This is a weak but useful model.")
else:
    print("The model does not explain much variance. Goals are hard to predict.")

# Print actual vs predicted for first 10 test samples
print("\nFirst 10 test predictions:")
comparison = pd.DataFrame({
    'Actual': y_test[:10].values,
    'Predicted': y_pred[:10].round(2)
})
print(comparison)