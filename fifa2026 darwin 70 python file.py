#nishant hamal section














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










#Aryan section










#Bishal section