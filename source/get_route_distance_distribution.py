import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data
singapore_rotterdam_df = pd.read_csv("data/marine_traffic_ais/singapore_rotterdam/singapore_rotterdam.csv")
rotterdam_singapore_df = pd.read_csv("data/marine_traffic_ais/singapore_rotterdam/rotterdam_singapore.csv")

# Step 1: Concatenate the two dataframes
combined_df = pd.concat([singapore_rotterdam_df, rotterdam_singapore_df], ignore_index=True)

# Step 2: Clean and filter invalid "Distance To Go" rows
combined_df["Distance To Go"] = pd.to_numeric(combined_df["Distance To Go"], errors="coerce")
combined_df["Voyage State"] = combined_df["Voyage State"].astype(str)

# Drop rows where 'Distance To Go' == 0 and 'Voyage State' is neither 'In port' nor 'Bunkering within Port'
mask = (combined_df["Distance To Go"] == 0) & (~combined_df["Voyage State"].isin(["In port", "Bunkering within Port", "In Origin Port"]))
filtered_df = combined_df[~mask].copy()

# Step 3: Add 'Voyage Distance' column
filtered_df["Distance Travelled"] = pd.to_numeric(filtered_df["Distance Travelled"], errors="coerce")
filtered_df["Voyage Distance"] = filtered_df["Distance Travelled"] + filtered_df["Distance To Go"]

# Step 4: Save filtered and enriched DataFrame to CSV
output_csv = "data/marine_traffic_ais/singapore_rotterdam/combined.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
filtered_df.to_csv(output_csv, index=False)
print(f"✔ Cleaned and combined data saved to '{output_csv}'")

# Plot the distribution
plt.figure(figsize=(5, 4))
filtered_df["Voyage Distance"].dropna().plot.hist(bins=30, edgecolor='black', color='skyblue')

# Compute statistics
average_voyage_distance = filtered_df["Voyage Distance"].mean()
std_voyage_distance = filtered_df["Voyage Distance"].std()

# Add vertical dashed line at the mean with formatted legend
plt.axvline(
    average_voyage_distance,
    color='red',
    linestyle='--',
    linewidth=2,
    label=f'Mean: {average_voyage_distance:.0f} nm\n±1 Std Dev: {std_voyage_distance:.0f} nm'
)

# Add semi-opaque band for ±1 std deviation
plt.axvspan(
    average_voyage_distance - std_voyage_distance,
    average_voyage_distance + std_voyage_distance,
    color='red',
    alpha=0.2
)

# Add titles and labels
#plt.title("Distribution of Voyage Distance (nm)", fontsize=16)
plt.xlabel("Voyage Distance (nautical miles)", fontsize=14)
plt.ylabel("Number of Voyages", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(loc='upper left')

# Save the plot
os.makedirs("plots", exist_ok=True)
plt.tight_layout()
plt.savefig("plots/voyage_distance_distribution.png", dpi=300)
plt.savefig("plots/voyage_distance_distribution.pdf")

print(f"Average voyage distance: {average_voyage_distance} +/- {std_voyage_distance}")
