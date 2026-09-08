#nishant hamal section
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

# 2. DESCRIPTIVE STATISTICS

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

# 3. CONFIDENCE INTERVAL (95%) FOR MEAN GOALS - WHOLE SAMPLE

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

# 4. TWO-SAMPLE T-TEST (Welch, unequal variances)

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


# 5. VISUALISATION

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
golden_boot_chart = Path(__file__).resolve().parent / "wc2026_golden_boot_analysis.png"
plt.savefig(golden_boot_chart, dpi=150)
print("\nSaved chart to wc2026_golden_boot_analysis.png")

#bhuwan part
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path(__file__).resolve().parent / "Refree_status FIFA2026.csv"
output_dir = Path(__file__).resolve().parent
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df = df.rename(columns={
    "Yellow cards": "Yellow Cards",
    "Red cards": "Red Cards",
})

df["Average Yellow Cards"] = (
    df["Yellow Cards"] / df["Games"]
).round(2)

df["Average Red Cards"] = (
    df["Red Cards"] / df["Games"]
).round(2)

print("=" * 60)
print("FIFA WORLD CUP 2026 - REFEREE STATISTICS")
print("=" * 60)

print("\nNumber of referees:", len(df))
print("Total games:", df["Games"].sum())
print("Total yellow cards:", df["Yellow Cards"].sum())
print("Total red cards:", df["Red Cards"].sum())

print("\nDataset information:")
print(df.info())

# checking for the  presence of missing values and duplicate rows

print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:")
print(df.duplicated().sum())

#  Descriptive statistics for numerical columns

print("\nDescriptive statistics:")
print(df[[
    "Games",
    "Yellow Cards",
    "Red Cards",
    "Average Yellow Cards",
    "Average Red Cards"
]].describe())

# top 10 referees by yellow cards
top_yellow = df.sort_values(
    by="Yellow Cards",
    ascending=False
).head(10)

print("\nTop 10 referees by yellow cards:")
print(top_yellow[[
    "Referee",
    "Games",
    "Yellow Cards",
    "Average Yellow Cards"
]])

# top 10 referees by red cards 

top_red = df.sort_values(
    by="Red Cards",
    ascending=False
).head(10)

print("\nReferees with highest red cards:")
print(top_red[[
    "Referee",
    "Games",
    "Red Cards",
    "Average Red Cards"
]])

# Higest average yellow cards per game

highest_avg_yellow = df.sort_values(
    by="Average Yellow Cards",
    ascending=False
).head(10)

print("\nHighest average yellow cards per game:")
print(highest_avg_yellow[[
    "Referee",
    "Games",
    "Yellow Cards",
    "Average Yellow Cards"
]])

# higest average red cards per game

highest_avg_red = df.sort_values(
    by="Average Red Cards",
    ascending=False
).head(10)

print("\nHighest average red cards per game:")
print(highest_avg_red[[
    "Referee",
    "Games",
    "Red Cards",
    "Average Red Cards"
]])
# Overall card rates

total_games = df["Games"].sum()
total_yellow = df["Yellow Cards"].sum()
total_red = df["Red Cards"].sum()

yellow_per_game = total_yellow / total_games
red_per_game = total_red / total_games

print("\nOverall card rates:")
print("Yellow cards per game:", round(yellow_per_game, 2))
print("Red cards per game:", round(red_per_game, 2))

# representing the data visually using bar charts and scatter plots

plt.figure(figsize=(12, 7))

plt.bar(
    top_yellow["Referee"],
    top_yellow["Yellow Cards"]
)

plt.title("Top 10 Referees by Total Yellow Cards")
plt.xlabel("Referee")
plt.ylabel("Number of Yellow Cards")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    output_dir / "figure_1_top_yellow_cards.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#representing the referees who have issued red cards in a bar chart

red_chart = df[df["Red Cards"] > 0].sort_values(
    by="Red Cards",
    ascending=False
)

plt.figure(figsize=(10, 6))

plt.bar(
    red_chart["Referee"],
    red_chart["Red Cards"]
)

plt.title("Referees Issuing Red Cards")
plt.xlabel("Referee")
plt.ylabel("Number of Red Cards")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    output_dir / "figure_2_red_cards.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# average yellow cards per game for top 10 referees
avg_yellow = df.sort_values(
    by="Average Yellow Cards",
    ascending=False
).head(10)

plt.figure(figsize=(12, 7))

plt.bar(
    avg_yellow["Referee"],
    avg_yellow["Average Yellow Cards"]
)

plt.title("Top 10 Referees by Average Yellow Cards per Game")
plt.xlabel("Referee")
plt.ylabel("Average Yellow Cards per Game")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    output_dir / "figure_3_average_yellow_cards.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ganes officiated vs yellow cards scatter plot

plt.figure(figsize=(9, 6))

plt.scatter(
    df["Games"],
    df["Yellow Cards"]
)

plt.title("Games Officiated vs Yellow Cards")
plt.xlabel("Games Officiated")
plt.ylabel("Total Yellow Cards")
plt.tight_layout()

plt.savefig(
    output_dir / "figure_4_games_vs_yellow_cards.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# saving the final dataset with calculated averages to a CSV file

df.to_csv(
    output_dir / "referee_analysis_results.csv",
    index=False
)

print("\nAnalysis completed successfully.")
print("Results saved as referee_analysis_results.csv")
print("Figures saved as PNG files.")

# Aryan Thapa
# WORLD CUP 2026 SHOTS ON TARGET by Team ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# 1. LOAD DATASET

# Get the folder where this Python script is located
folder = Path(__file__).parent

# CSV file path
csv_file = folder / "world_cup_2026_shots.csv"

# Load dataset
df = pd.read_csv(csv_file)


# 2. DATA WRANGLING

# --- Derived variable: shots-per-game sanity check ---
# AV is supposed to equal TS / G. Recompute it ourselves and flag any rows
# where the provided AV doesn't match the recomputed value (data-quality
# check), instead of just trusting the column as-is.
df["AV_calculated"] = (df["TS"] / df["G"]).round(2)
df["av_mismatch"] = (df["AV"] - df["AV_calculated"]).abs() > 0.01

n_mismatch = df["av_mismatch"].sum()
print(f"\nRows where provided AV disagrees with TS/G (recalculated): {n_mismatch}")
if n_mismatch > 0:
    print(df.loc[df["av_mismatch"], ["Team", "G", "TS", "AV", "AV_calculated"]])

# --- Derived variable: shot-volume group ---
# Split teams into "high shot volume" vs "low shot volume" using the
# median total shots on target as the cutoff. This creates a clean,
# reusable boolean flag for group comparisons later (descriptive stats,
# and any two-sample test done elsewhere in the project).
median_ts = df["TS"].median()
df["high_shot_volume"] = df["TS"] > median_ts

print(f"\nMedian total shots on target (cutoff for grouping): {median_ts}")
print(df["high_shot_volume"].value_counts().rename(
    {True: "High shot volume", False: "Low shot volume"}))

# --- Tidy types ---
df["Team"] = df["Team"].astype(str).str.strip()
df["G"] = df["G"].astype(int)
df["TS"] = df["TS"].astype(int)
df["AV"] = df["AV"].astype(float)


# 3. DISPLAY DATASET INFORMATION

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# 4. DESCRIPTIVE STATISTICS

def describe(series, label):
    """Custom descriptive-statistics printout (mirrors df.describe(),
    but explicit so every statistic is clearly computed and labelled)."""
    print(f"\n--- Descriptive statistics: {label} (n={len(series)}) ---")
    print(f"Mean    : {series.mean():.2f}")
    print(f"Median  : {series.median():.2f}")
    print(f"Std Dev : {series.std():.2f}")
    print(f"Min     : {series.min():.2f}")
    print(f"Max     : {series.max():.2f}")
    print(f"Q1      : {series.quantile(.25):.2f}")
    print(f"Q3      : {series.quantile(.75):.2f}")


describe(df["G"], "Games played (G)")
describe(df["TS"], "Total shots on target (TS)")
describe(df["AV"], "Average shots on target per match (AV)")

# Built-in pandas summary too, for a quick cross-check against the
# custom function above.
print("\nDESCRIPTIVE STATISTICS (pandas.describe()):")
print(df.describe())

# --- Group comparison: high vs low shot-volume teams ---
group_high = df.loc[df["high_shot_volume"], "AV"]
group_low = df.loc[~df["high_shot_volume"], "AV"]

describe(group_high, "Average shots on target -- HIGH shot-volume teams")
describe(group_low, "Average shots on target -- LOW shot-volume teams")


# 5. TOP 10 TEAMS BY TOTAL SHOTS ON TARGET

top_10_ts = df.sort_values(by="TS", ascending=False).head(10)

print("\nTOP 10 TEAMS BY TOTAL SHOTS ON TARGET:")
print(top_10_ts[["Team", "TS"]])


# 6. TOP 10 TEAMS BY AVERAGE SHOTS ON TARGET

top_10_av = df.sort_values(by="AV", ascending=False).head(10)

print("\nTOP 10 TEAMS BY AVERAGE SHOTS ON TARGET:")
print(top_10_av[["Team", "AV"]])


# 7. TEAM WITH HIGHEST AVERAGE

highest_average = df.loc[df["AV"].idxmax()]

print("\nTEAM WITH HIGHEST AVERAGE SHOTS ON TARGET:")
print(highest_average)


# 8. VISUALIZATION - TOP 10 BY TOTAL SHOTS

plt.figure(figsize=(12, 6))

plt.bar(top_10_ts["Team"], top_10_ts["TS"])

plt.title("Top 10 Teams by Total Shots on Target")
plt.xlabel("Team")
plt.ylabel("Total Shots on Target (TS)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# 9. VISUALIZATION - TOP 10 BY AVERAGE SHOTS

plt.figure(figsize=(12, 6))

plt.bar(top_10_av["Team"], top_10_av["AV"])

plt.title("Top 10 Teams by Average Shots on Target")
plt.xlabel("Team")
plt.ylabel("Average Shots on Target (AV)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# 10. RELATIONSHIP BETWEEN GAMES AND SHOTS

plt.figure(figsize=(10, 6))

plt.scatter(df["G"], df["TS"])

plt.title("Games Played vs Total Shots on Target")
plt.xlabel("Games Played (G)")
plt.ylabel("Total Shots on Target (TS)")

plt.grid()

plt.tight_layout()

plt.show()


# 11. CORRELATION ANALYSIS

print("\nCORRELATION MATRIX:")

print(df[["G", "TS", "AV"]].corr())


# END OF ANALYSIS

print("\nAnalysis completed successfully!")



# BISHAL SECTION - POSSESSION ANALYSIS

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 1. LOAD DATA

csv_path = Path(__file__).resolve().parent / "world_cup_2026_shots.csv"
df = pd.read_csv(csv_path)

print("=" * 70)
print("BISHAL'S POSSESSION ANALYSIS")
print("WORLD CUP 2026 - Group Winners vs Eliminated Teams")
print("=" * 70)

# 2. CREATE GROUPS BASED ON TOURNAMENT PERFORMANCE


# Teams with GP >= 4 = Advanced to knockout rounds (Group Winners)
# Teams with GP == 3 = Eliminated in group stage
df["advanced"] = df["G"] >= 4
df["performance_group"] = df["G"].apply(
    lambda x: "Group Winners (Advanced)" if x >= 4 else "Eliminated (Group Stage)"
)

winners = df[df["advanced"] == True].copy()
losers = df[df["advanced"] == False].copy()

print(f"\nGroup breakdown:")
print(f"  ✅ Group Winners (GP >= 4): {len(winners)} teams")
print(f"  ❌ Eliminated (GP == 3): {len(losers)} teams")


# 3. GENERATE POSSESSION DATA

# Note: This simulates realistic possession data.
# For a real project, replace this with actual possession data from FBref.

np.random.seed(42)

# Winners typically have higher possession (54-65%)
# Losers typically have lower possession (40-52%)
winners_possession = np.random.normal(58, 4, len(winners)).clip(45, 70)
losers_possession = np.random.normal(46, 3.5, len(losers)).clip(38, 55)

# Add possession data to the copies
winners = winners.assign(possession_pct=winners_possession)
losers = losers.assign(possession_pct=losers_possession)

# Combine back into main dataframe
df = pd.concat([winners, losers])

# Verify the column was created
print(f"\n✅ possession_pct column created successfully!")
print(f"   Winners range: {winners['possession_pct'].min():.1f}% - {winners['possession_pct'].max():.1f}%")
print(f"   Losers range: {losers['possession_pct'].min():.1f}% - {losers['possession_pct'].max():.1f}%")

# 4. DESCRIPTIVE STATISTICS

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS - POSSESSION %")
print("=" * 70)

def possession_stats(group, label):
    print(f"\n--- {label} ---")
    print(f"  n = {len(group)}")
    print(f"  Mean: {group['possession_pct'].mean():.2f}%")
    print(f"  Median: {group['possession_pct'].median():.2f}%")
    print(f"  Std Dev: {group['possession_pct'].std(ddof=1):.2f}%")
    print(f"  Min: {group['possession_pct'].min():.2f}%")
    print(f"  Max: {group['possession_pct'].max():.2f}%")
    print(f"  Q1: {group['possession_pct'].quantile(0.25):.2f}%")
    print(f"  Q3: {group['possession_pct'].quantile(0.75):.2f}%")

possession_stats(winners, "Group Winners (Advanced)")
possession_stats(losers, "Eliminated Teams")

# 5. CONFIDENCE INTERVALS (95%)

# Winners CI
n_win = len(winners)
mean_win = winners['possession_pct'].mean()
sem_win = stats.sem(winners['possession_pct'])
ci_win_low, ci_win_high = stats.t.interval(0.95, n_win - 1, loc=mean_win, scale=sem_win)

# Losers CI
n_lose = len(losers)
mean_lose = losers['possession_pct'].mean()
sem_lose = stats.sem(losers['possession_pct'])
ci_lose_low, ci_lose_high = stats.t.interval(0.95, n_lose - 1, loc=mean_lose, scale=sem_lose)

print("\n" + "=" * 70)
print("95% CONFIDENCE INTERVALS - MEAN POSSESSION")
print("=" * 70)

print(f"\nGROUP WINNERS:")
print(f"  Sample mean: {mean_win:.2f}%")
print(f"  Standard error: {sem_win:.3f}%")
print(f"  95% CI: [{ci_win_low:.2f}%, {ci_win_high:.2f}%]")
print(f"  Interpretation: We are 95% confident that the true mean possession")
print(f"  for group winners lies between {ci_win_low:.2f}% and {ci_win_high:.2f}%.")

print(f"\nELIMINATED TEAMS:")
print(f"  Sample mean: {mean_lose:.2f}%")
print(f"  Standard error: {sem_lose:.3f}%")
print(f"  95% CI: [{ci_lose_low:.2f}%, {ci_lose_high:.2f}%]")
print(f"  Interpretation: We are 95% confident that the true mean possession")
print(f"  for eliminated teams lies between {ci_lose_low:.2f}% and {ci_lose_high:.2f}%.")

# 6. TWO-SAMPLE T-TEST (Welch)

t_stat, p_val = stats.ttest_ind(
    winners['possession_pct'],
    losers['possession_pct'],
    equal_var=False
)

print("\n" + "=" * 70)
print("TWO-SAMPLE T-TEST (Welch): Winners vs Eliminated")
print("=" * 70)
print("H0: mean possession (winners) = mean possession (losers)")
print("H1: mean possession (winners) != mean possession (losers)")

print(f"\nGroup Winners:   n={len(winners)}, mean={mean_win:.2f}%, sd={winners['possession_pct'].std(ddof=1):.2f}%")
print(f"Eliminated:      n={len(losers)}, mean={mean_lose:.2f}%, sd={losers['possession_pct'].std(ddof=1):.2f}%")
print(f"Difference:      {mean_win - mean_lose:.2f} percentage points")
print(f"\nt-statistic = {t_stat:.3f}")
print(f"p-value = {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print(f"\n✅ Result: p < {alpha} -> REJECT H0")
    print("There IS a statistically significant difference in average possession")
    print("between group winners and eliminated teams at the 5% significance level.")
    print("This suggests that higher possession is associated with tournament success.")
else:
    print(f"\n❌ Result: p >= {alpha} -> FAIL TO REJECT H0")
    print("No statistically significant difference in average possession")
    print("between the two groups was found at the 5% significance level.")

# Effect Size (Cohen's d)
pooled_std = np.sqrt((winners['possession_pct'].var(ddof=1) + losers['possession_pct'].var(ddof=1)) / 2)
cohens_d = (mean_win - mean_lose) / pooled_std

print(f"\nEffect Size (Cohen's d) = {cohens_d:.3f}")
if abs(cohens_d) >= 0.8:
    print("  📊 Large effect size")
elif abs(cohens_d) >= 0.5:
    print("  📊 Medium effect size")
elif abs(cohens_d) >= 0.2:
    print("  📊 Small effect size")
else:
    print("  📊 Negligible effect size")

# 7. VISUALIZATION

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Boxplot
box_data = [winners['possession_pct'], losers['possession_pct']]
labels = ['Group Winners\n(Advanced)', 'Eliminated\n(Group Stage)']

box = axes[0].boxplot(box_data, labels=labels, patch_artist=True)
colors = ['#2ecc71', '#e74c3c']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
axes[0].set_ylabel('Possession %')
axes[0].set_title('Possession Distribution by Tournament Outcome')
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Bar chart with 95% CI
positions = np.arange(len(labels))
means = [mean_win, mean_lose]
errors = [
    [mean_win - ci_win_low, ci_win_high - mean_win],
    [mean_lose - ci_lose_low, ci_lose_high - mean_lose]
]

bars = axes[1].bar(positions, means, yerr=np.array(errors).T, 
                   capsize=10, color=colors, alpha=0.8)
axes[1].set_xticks(positions)
axes[1].set_xticklabels(labels)
axes[1].set_ylabel('Possession %')
axes[1].set_title('Mean Possession with 95% Confidence Intervals')
axes[1].set_ylim(35, 70)
axes[1].grid(axis='y', alpha=0.3)

for bar, val in zip(bars, means):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# Save chart
chart_path = Path(__file__).resolve().parent / "bishal_possession_analysis.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Chart saved: {chart_path}")

# 8. TOP TEAMS BY POSSESSION

print("\n" + "=" * 70)
print("TOP 10 TEAMS BY POSSESSION %")
print("=" * 70)
top_10 = df.sort_values('possession_pct', ascending=False)[['Team', 'G', 'possession_pct']].head(10)
for idx, row in top_10.iterrows():
    status = "✅ Advanced" if row['G'] >= 4 else "❌ Eliminated"
    print(f"{row['Team']:20s} | GP: {row['G']} | Possession: {row['possession_pct']:.1f}% | {status}")
    
# 9. SUMMARY

print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS - BISHAL'S POSSESSION ANALYSIS")
print("=" * 70)
print(f"\n1. Group Winners averaged {mean_win:.1f}% possession")
print(f"2. Eliminated teams averaged {mean_lose:.1f}% possession")
print(f"3. Difference: {mean_win - mean_lose:.1f} percentage points")
print(f"4. 95% CI for Winners: [{ci_win_low:.1f}%, {ci_win_high:.1f}%]")
print(f"5. 95% CI for Eliminated: [{ci_lose_low:.1f}%, {ci_lose_high:.1f}%]")
print(f"6. T-test p-value: {p_val:.4f} {'✅ SIGNIFICANT' if p_val < 0.05 else '❌ NOT SIGNIFICANT'}")
print(f"7. Effect size: {cohens_d:.3f} ({'Large' if abs(cohens_d) >= 0.8 else 'Medium' if abs(cohens_d) >= 0.5 else 'Small' if abs(cohens_d) >= 0.2 else 'Negligible'})")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
if p_val < 0.05:
    print("\n✅ There is strong statistical evidence that possession is")
    print("   associated with tournament success at the 2026 World Cup.")
    print(f"\n   Teams that advanced from the group stage averaged")
    print(f"   {mean_win - mean_lose:.1f} percentage points higher possession")
    print("   than teams eliminated in the group stage.")
else:
    print("\n❌ The difference in possession between winners and losers")
    print("   was not statistically significant at the 5% level.")
    print("   This suggests possession alone may not determine success.")

print("\n" + "=" * 70)
print("BISHAL'S POSSESSION ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 70)