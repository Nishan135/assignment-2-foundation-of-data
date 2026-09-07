import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. DATA WRANGLING
# ---------------------------------------------------------------
df = pd.read_csv("C:\\Users\\Nishan hamal\\Desktop\\foundation of data\\wc2026_golden_boot_scorers.csv")

# Derived variable: scoring efficiency
df["minutes_per_goal"] = df["minutes_played"] / df["goals"]

# Tidy boolean flag
df["semifinalist_team"] = df["semifinalist_team"].map({"Yes": True, "No": False})

print("=" * 70)
print("SAMPLE (Golden Boot contenders, goals >= 3), n =", len(df))
print("=" * 70)
print(df.to_string(index=False))

group_semi = df.loc[df["semifinalist_team"], "goals"]
group_non_semi = df.loc[~df["semifinalist_team"], "goals"]

# ---------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------
def describe(series, label):
    print(f"\n--- Descriptive statistics: {label} (n={len(series)}) ---")
    print(f"Mean goals   : {series.mean():.2f}")
    print(f"Median goals : {series.median():.2f}")
    print(f"Std dev      : {series.std(ddof=1):.2f}")
    print(f"Min / Max    : {series.min()} / {series.max()}")
    print(f"IQR          : {series.quantile(.25):.2f} - {series.quantile(.75):.2f}")

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)
describe(df["goals"], "All Golden Boot contenders")
describe(group_semi, "Players from semi-finalist teams")
describe(group_non_semi, "Players from non-semi-finalist teams")

print("\n--- Minutes-per-goal (efficiency), whole sample ---")
print(df["minutes_per_goal"].describe().round(2))

# ---------------------------------------------------------------
# 3. CONFIDENCE INTERVAL (95%) FOR MEAN GOALS - WHOLE SAMPLE
# ---------------------------------------------------------------
n = len(df)
mean_goals = df["goals"].mean()
sem_goals = stats.sem(df["goals"])  # standard error of the mean
ci_low, ci_high = stats.t.interval(confidence=0.95, df=n - 1,
                                    loc=mean_goals, scale=sem_goals)

print("\n" + "=" * 70)
print("95% CONFIDENCE INTERVAL - mean goals scored by a Golden Boot")
print("contender in the WC2026 goalscorer population")
print("=" * 70)
print(f"Sample mean       : {mean_goals:.2f} goals")
print(f"Standard error     : {sem_goals:.3f}")
print(f"95% CI             : [{ci_low:.2f}, {ci_high:.2f}] goals")
print("Interpretation: we are 95% confident the true mean number of goals")
print("scored by a member of the WC2026 'leading scorers' population lies")
print(f"between {ci_low:.2f} and {ci_high:.2f} goals.")

# ---------------------------------------------------------------
# 4. TWO-SAMPLE T-TEST (Welch, unequal variances)
# ---------------------------------------------------------------
t_stat, p_val = stats.ttest_ind(group_semi, group_non_semi, equal_var=False)

print("\n" + "=" * 70)
print("TWO-SAMPLE T-TEST (Welch): semi-finalist-team vs other players")
print("=" * 70)
print("H0: mean goals (semi-finalist team) = mean goals (other teams)")
print("H1: mean goals (semi-finalist team) != mean goals (other teams)")
print(f"Semi-finalist team players   : n={len(group_semi)}, mean={group_semi.mean():.2f}, sd={group_semi.std(ddof=1):.2f}")
print(f"Non-semi-finalist team players: n={len(group_non_semi)}, mean={group_non_semi.mean():.2f}, sd={group_non_semi.std(ddof=1):.2f}")
print(f"t-statistic = {t_stat:.3f}")
print(f"p-value     = {p_val:.4f}")
alpha = 0.05
if p_val < alpha:
    print(f"Result: p < {alpha} -> reject H0. There IS a statistically significant")
    print("difference in average goals scored between the two groups.")
else:
    print(f"Result: p >= {alpha} -> fail to reject H0. No statistically significant")
    print("difference in average goals scored between the two groups was found")
    print("at the 5% significance level.")

# Golden Boot winner call-out (top of the sample by goals, tie-break assists)
winner = df.sort_values(["goals", "assists"], ascending=False).iloc[0]
print("\n" + "=" * 70)
print("GOLDEN BOOT WINNER (top scorer in the dataset)")
print("=" * 70)
print(f"{winner['player']} ({winner['team']}) - {winner['goals']} goals, "
      f"{winner['assists']} assists in {winner['appearances']} appearances")

# ---------------------------------------------------------------
# 5. VISUALISATION
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart of top scorers
top_sorted = df.sort_values("goals", ascending=True)
axes[0].barh(top_sorted["player"], top_sorted["goals"],
             color=["#c8102e" if s else "#4c72b0" for s in top_sorted["semifinalist_team"]])
axes[0].set_xlabel("Goals scored")
axes[0].set_title("WC2026 Golden Boot contenders\n(red = semi-finalist team)")

# Boxplot comparison
axes[1].boxplot([group_semi, group_non_semi], tick_labels=["Semi-finalist\nteam", "Other\nteam"])
axes[1].set_ylabel("Goals scored")
axes[1].set_title("Goals scored: semi-finalist vs other teams")

plt.tight_layout()
plt.savefig("wc2026_golden_boot_analysis.png", dpi=150)
print("\nSaved chart to wc2026_golden_boot_analysis.png")
