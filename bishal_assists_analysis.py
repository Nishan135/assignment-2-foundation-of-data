
# Bishal Basnet -Section

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. DATA WRANGLING

csv_path = Path(__file__).resolve().parent / "wc2026_golden_boot_scorers.csv"
df = pd.read_csv(csv_path)

# Clean data
df["semifinalist_team"] = df["semifinalist_team"].map({"Yes": True, "No": False})

# Create derived variables
df["assists_per_game"] = df["assists"] / df["appearances"]
df["goal_involvements"] = df["goals"] + df["assists"]

print("=" * 70)
print("DATASET OVERVIEW - ASSISTS ANALYSIS")
print("=" * 70)
print(f"Total players: {len(df)}")
print(df[["player", "team", "position", "goals", "assists"]].head())


# 2. DATA PREPARATION - TWO GROUPS

# Separate into midfielders and forwards
midfielders = df[df["position"] == "Midfielder"]
forwards = df[df["position"] == "Forward"]

print(f"\nMidfielders (n={len(midfielders)}): {midfielders['player'].tolist()}")
print(f"Forwards (n={len(forwards)}): {forwards['player'].tolist()}")


# 3. DESCRIPTIVE STATISTICS

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS - ASSISTS")
print("=" * 70)

def describe_group(group, label):
    print(f"\n--- {label} (n={len(group)}) ---")
    print(f"Mean assists: {group['assists'].mean():.2f}")
    print(f"Median assists: {group['assists'].median():.2f}")
    print(f"Std deviation: {group['assists'].std(ddof=1):.2f}")
    print(f"Min/Max: {group['assists'].min()} / {group['assists'].max()}")
    print(f"Mean per game: {group['assists_per_game'].mean():.2f}")

describe_group(midfielders, "Midfielders")
describe_group(forwards, "Forwards")

print(f"\n--- All Players ---")
print(f"Mean assists: {df['assists'].mean():.2f}")
print(f"Total assists: {df['assists'].sum()}")


# 4. CONFIDENCE INTERVAL (95%)

# Forwards
n_fwd = len(forwards)
mean_fwd = forwards["assists"].mean()
sem_fwd = stats.sem(forwards["assists"])
ci_fwd_low, ci_fwd_high = stats.t.interval(0.95, n_fwd - 1, loc=mean_fwd, scale=sem_fwd)

# Midfielders
n_mid = len(midfielders)
mean_mid = midfielders["assists"].mean()
sem_mid = stats.sem(midfielders["assists"])
ci_mid_low, ci_mid_high = stats.t.interval(0.95, n_mid - 1, loc=mean_mid, scale=sem_mid)

print("\n" + "=" * 70)
print("95% CONFIDENCE INTERVALS - MEAN ASSISTS")
print("=" * 70)
print(f"FORWARDS:")
print(f"  Sample mean: {mean_fwd:.2f}")
print(f"  95% CI: [{ci_fwd_low:.2f}, {ci_fwd_high:.2f}]")
print(f"  Interpretation: We are 95% confident the true mean assists for forwards")
print(f"  lies between {ci_fwd_low:.2f} and {ci_fwd_high:.2f}.")

print(f"\nMIDFIELDERS:")
print(f"  Sample mean: {mean_mid:.2f}")
print(f"  95% CI: [{ci_mid_low:.2f}, {ci_mid_high:.2f}]")
print(f"  Interpretation: We are 95% confident the true mean assists for midfielders")
print(f"  lies between {ci_mid_low:.2f} and {ci_mid_high:.2f}.")

# 5. TWO-SAMPLE T-TEST (Welch's t-test)


t_stat, p_val = stats.ttest_ind(forwards["assists"], midfielders["assists"], equal_var=False)

print("\n" + "=" * 70)
print("TWO-SAMPLE T-TEST (Forwards vs Midfielders)")
print("=" * 70)
print("H0: mean assists (forwards) = mean assists (midfielders)")
print("H1: mean assists (forwards) != mean assists (midfielders)")
print(f"Forwards: n={n_fwd}, mean={mean_fwd:.2f}, sd={forwards['assists'].std(ddof=1):.2f}")
print(f"Midfielders: n={n_mid}, mean={mean_mid:.2f}, sd={midfielders['assists'].std(ddof=1):.2f}")
print(f"t-statistic = {t_stat:.3f}")
print(f"p-value = {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print(f"\nResult: p < {alpha} -> REJECT H0")
    print("There IS a statistically significant difference in average assists")
    print("between forwards and midfielders at the 5% significance level.")
else:
    print(f"\nResult: p >= {alpha} -> FAIL TO REJECT H0")
    print("No statistically significant difference in average assists")
    print("between forwards and midfielders was found at the 5% significance level.")


# 6. VISUALISATION - CHART

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Bar chart
positions = ["Forwards", "Midfielders"]
means = [mean_fwd, mean_mid]
colors = ["#1f77b4", "#ff7f0e"]

bars = axes[0].bar(positions, means, color=colors, width=0.6)
axes[0].set_ylabel("Mean Assists")
axes[0].set_title("Mean Assists: Forwards vs Midfielders")
axes[0].set_ylim(0, max(means) + 1)

for bar, val in zip(bars, means):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

# Plot 2: Boxplot
box_data = [forwards["assists"], midfielders["assists"]]
box = axes[1].boxplot(box_data, labels=positions, patch_artist=True)
axes[1].set_ylabel("Assists")
axes[1].set_title("Assists Distribution by Position")

for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

plt.tight_layout()

# Save the chart
output_path = Path(__file__).resolve().parent / "bishal_assists_comparison.png"
plt.savefig(output_path, dpi=150)
print(f"\nChart saved: {output_path}")


# 7. TOP ASSIST PROVIDERS

print("\n" + "=" * 70)
print("TOP 5 ASSIST PROVIDERS")
print("=" * 70)
top_assists = df.sort_values("assists", ascending=False).head(5)
print(top_assists[["player", "team", "position", "assists"]].to_string(index=False))


# 8. SUMMARY OF FINDINGS

print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)
print(f"1. Forwards average {mean_fwd:.2f} assists")
print(f"2. Midfielders average {mean_mid:.2f} assists")
print(f"3. 95% CI for forwards: [{ci_fwd_low:.2f}, {ci_fwd_high:.2f}]")
print(f"4. 95% CI for midfielders: [{ci_mid_low:.2f}, {ci_mid_high:.2f}]")
print(f"5. T-test p-value: {p_val:.4f}")

print("\nAnalysis completed successfully!")