import pandas as pd
import glob
import os

# Directory containing the CSV files
input_dir = 'data/marine_traffic_ais/'
output_file = os.path.join(input_dir, 'ROT_departures_operational_data.csv')

# Use glob to find all relevant files
csv_files = glob.glob(os.path.join(input_dir, 'MarineTraffic_Vessels_Export_2025-05-21(*).csv'))

# Read and combine all CSVs
df_list = [pd.read_csv(f) for f in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Remove duplicates based on 'Imo' column
deduped_df = combined_df.drop_duplicates(subset='Imo')

# Save the cleaned DataFrame
deduped_df.to_csv(output_file, index=False)

print(f"Combined and cleaned data saved to {output_file}")

