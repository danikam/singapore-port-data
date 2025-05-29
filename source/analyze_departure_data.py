import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import requests
from time import sleep
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import seaborn as sns
import argparse
import numpy as np

# Argument parser for command-line options
parser = argparse.ArgumentParser(description="Analyze vessel departures from Singapore.")
parser.add_argument('--api_key', required=True, help='API key for accessing MPA vessel particulars API')
args = parser.parse_args()

API_KEY = args.api_key
PARTICULARS_URL = "https://sg-mdh-api.mpa.gov.sg/v1/vessel/particulars/imonumber/{}"

def export_all_departures_to_csv(input_json="data/vessel_departures.json", output_csv="output/all_departures.csv"):
    if os.path.exists(output_csv):
        print(f"All departures CSV already exists at '{output_csv}', skipping.")
        return pd.read_csv(output_csv)

    with open(input_json, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Parse arrival and departure times
    df["arrival_datetime"] = pd.to_datetime(df["reportedArrivalTime"], errors="coerce")
    df["departure_datetime"] = pd.to_datetime(df["reportedDepartureTime"], errors="coerce")

    # Compute port duration in days
    df["PortDuration"] = (df["departure_datetime"] - df["arrival_datetime"]).dt.total_seconds() / (24 * 3600)

    # Extract hour bucket for deduplication
    df["departure_hour"] = df["departure_datetime"].dt.floor("H")

    # Extract IMO number
    df["imoNumber"] = df["vesselParticulars"].apply(
        lambda x: x.get("imoNumber") if isinstance(x, dict) else None
    ).astype(str)

    # Drop duplicates: same IMO number within the same hour
    before = len(df)
    df = df.drop_duplicates(subset=["imoNumber", "departure_hour"])
    after = len(df)

    print(f"Dropped {before - after} duplicates based on IMO number and hour-binned departure time.")

    # Clean up temporary columns
    df = df.drop(columns=["departure_hour"])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"All departures saved to '{output_csv}' with PortDuration column included.")
    return df


def filter_departures(df,
    #output_csv="output/LA_ROT_departures.csv",
    output_csv="output/ROT_departures.csv",
    #target_ports={"ROTTERDAM", "LOS ANGELES, CA", "LONG BEACH, CA"}
    target_ports={"ROTTERDAM"}
):
    if os.path.exists(output_csv):
        print(f"Output already exists at '{output_csv}', skipping.")
        return pd.read_csv(output_csv)

    df["nextPort_clean"] = df["nextPort"].str.upper().str.strip()
    filtered_df = df[df["nextPort_clean"].isin(target_ports)].drop(columns=["nextPort_clean"])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered departures saved to '{output_csv}'")
    return filtered_df

def fetch_particulars(imo):
    url = PARTICULARS_URL.format(imo)
    try:
        response = requests.get(url, headers={"apikey": API_KEY})
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            vessel_info = data[0]
            return {
                "imoNumber": imo,
                "grossTonnage": vessel_info.get("grossTonnage"),
                "netTonnage": vessel_info.get("netTonnage"),
                "deadweight": vessel_info.get("deadweight"),
                "vesselType": vessel_info.get("vesselType")
            }
    except Exception as e:
        print(f"Error for IMO {imo}: {e}")
    return {
        "imoNumber": imo,
        "grossTonnage": None,
        "netTonnage": None,
        "deadweight": None,
        "vesselType": None
    }


def add_vessel_particulars(df,
    #output_csv="output/LA_ROT_departures_enriched.csv",
    output_csv="output/ROT_departures_enriched.csv",
    max_workers=5
):
    if os.path.exists(output_csv):
        print(f"Enriched CSV already exists at '{output_csv}', skipping API calls.")
        return pd.read_csv(output_csv)

    # Convert vesselParticulars from string to dict if needed
    if isinstance(df["vesselParticulars"].iloc[0], str):
        df["vesselParticulars"] = df["vesselParticulars"].apply(ast.literal_eval)

    df["imoNumber"] = df["vesselParticulars"].apply(lambda x: x.get("imoNumber"))
    df["original_index"] = df.index  # track original rows

    unique_imos = df["imoNumber"].dropna().unique().tolist()
    print(f"Fetching vessel particulars for {len(unique_imos)} unique vessels using {max_workers} threads...")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_imo = {executor.submit(fetch_particulars, imo): imo for imo in unique_imos}
        for i, future in enumerate(as_completed(future_to_imo), 1):
            result = future.result()
            results.append(result)
            print(f"✔ Processed {i}/{len(unique_imos)} (IMO: {result['imoNumber']})", end="\r")

    results_df = pd.DataFrame(results).set_index("imoNumber")

    # Merge using a lookup (without exploding rows)
    df["grossTonnage"] = df["imoNumber"].map(results_df["grossTonnage"])
    df["netTonnage"] = df["imoNumber"].map(results_df["netTonnage"])
    df["deadweight"] = df["imoNumber"].map(results_df["deadweight"])
    df["vesselType"] = df["imoNumber"].map(results_df["vesselType"])

    df = df.drop(columns=["original_index"])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✔ Enriched DataFrame saved to '{output_csv}'")

    return df


def update_deadweight_from_csv(df,
    update_csv_path="data/missing_dwt_info.csv",
    #output_csv="output/LA_ROT_departures_updated.csv"
    output_csv="output/ROT_departures_updated.csv"
):
    if not os.path.exists(update_csv_path):
        print(f"File not found: {update_csv_path}")
        return df

    override_df = pd.read_csv(update_csv_path)

    # Ensure consistent types for join
    override_df["imoNumber"] = override_df["imoNumber"].astype(str)
    df["imoNumber"] = df["imoNumber"].astype(str)

    # Merge deadweight overrides by IMO number
    overrides = override_df[["imoNumber", "deadweight"]].set_index("imoNumber")
    original_len = len(df)

    # Update deadweight using map from overrides
    df = df.set_index("imoNumber")
    updated = df.index.intersection(overrides.index)
    df.loc[updated, "deadweight"] = overrides.loc[updated, "deadweight"]
    df = df.reset_index()

    print(f"Overwrote deadweight for {len(updated)} vessel(s) using {update_csv_path}")
    df.to_csv(output_csv, index=False)
    print(f"\nEnriched DataFrame saved to '{output_csv}'")
    return df

def plot_departures_per_day(all_df, filtered_df):
    if "reportedDepartureTime" not in all_df.columns or "reportedDepartureTime" not in filtered_df.columns:
        print("⚠ 'reportedDepartureTime' column not found in one or both dataframes.")
        return

    # Parse datetime fields
    all_df["departure_datetime"] = pd.to_datetime(all_df["reportedDepartureTime"], errors="coerce")
    filtered_df["departure_datetime"] = pd.to_datetime(filtered_df["reportedDepartureTime"], errors="coerce")

    # Floor to day and count
    all_counts = all_df["departure_datetime"].dt.floor("d").value_counts().sort_index()
    filtered_counts = filtered_df["departure_datetime"].dt.floor("d").value_counts().sort_index()

    # Unified time range
    if all_counts.empty and filtered_counts.empty:
        print("No valid datetime values to plot.")
        return

    all_dates = pd.date_range(
        start=min(all_counts.index.min(), filtered_counts.index.min()),
        end=max(all_counts.index.max(), filtered_counts.index.max()),
        freq="D"
    )

    # Reindex to fill gaps
    all_counts = all_counts.reindex(all_dates, fill_value=0)
    filtered_counts = filtered_counts.reindex(all_dates, fill_value=0)

    # Plot with secondary y-axis
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(all_counts.index, all_counts.values, color="gray", label="All Destinations", linewidth=1.5)
    ax1.set_ylabel("All Departures", color="gray", fontsize=20)
    ax1.tick_params(axis='y', labelcolor="gray")

    ax2 = ax1.twinx()
    #ax2.plot(filtered_counts.index, filtered_counts.values, color="darkorange", label="ROTTERDAM / LA / LONG BEACH", linewidth=1.5)
    ax2.plot(filtered_counts.index, filtered_counts.values, color="darkorange", label="To ROTTERDAM", linewidth=1.5)
    #ax2.set_ylabel("Rotterdam / LA / Long Beach Departures", color="darkorange", fontsize=19)
    ax2.set_ylabel("Rotterdam Departures", color="darkorange", fontsize=19)
    ax2.tick_params(axis='y', labelcolor="darkorange")

    ax1.set_title("Daily Vessel Departures: All Ports vs Selected Ports", fontsize=24)
    ax1.tick_params(axis='x', rotation=30)
    ax1.tick_params(axis='both', labelsize=18)
    ax2.tick_params(axis='both', labelsize=18)
    plt.grid(True, linestyle="--", axis='both', linewidth=0.5, alpha=0.7)
    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/departures_per_day_comparison.png", dpi=300)
    plt.savefig("plots/departures_per_day_comparison.pdf")
    print(f"Saved comparative departures-per-day plot (with dual y-axes) to plots/departures_per_day_comparison.png")
    # plt.show()
    
def plot_departures_by_vessel_class(all_df, classified_df):
    if "reportedDepartureTime" not in all_df.columns or "reportedDepartureTime" not in classified_df.columns:
        print("'reportedDepartureTime' column not found.")
        return
    if "vesselType_clean" not in classified_df.columns:
        print("'vesselType_clean' column not found in classified_df.")
        return

    # Parse datetime and floor to day
    all_df["departure_datetime"] = pd.to_datetime(all_df["reportedDepartureTime"], errors="coerce")
    classified_df["departure_datetime"] = pd.to_datetime(classified_df["reportedDepartureTime"], errors="coerce")
    classified_df["departure_date"] = classified_df["departure_datetime"].dt.floor("d")

    # Valid vessel types
    vessel_classes = ["Bulk Carrier", "Container Ship", "Tanker"]
    df = classified_df[classified_df["vesselType_clean"].isin(vessel_classes)]

    # Count per day per class
    grouped = df.groupby(["departure_date", "vesselType_clean"]).size().unstack(fill_value=0)

    # Full date range based on all_df
    full_range = pd.date_range(
        start=all_df["departure_datetime"].min().floor("d"),
        end=all_df["departure_datetime"].max().floor("d"),
        freq="D"
    )
    grouped = grouped.reindex(full_range, fill_value=0)

    # Plot
    plt.figure(figsize=(14, 6))
    for cls in vessel_classes:
        if cls in grouped.columns:
            plt.plot(grouped.index, grouped[cls], label=cls)

    plt.title("Daily Vessel Departures by Vessel Class", fontsize=24)
    #plt.xlabel("Date")
    plt.ylabel("Departures", fontsize=20)
    plt.legend(fontsize=18)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tick_params(axis='both', labelsize=18)
    plt.tick_params(axis='x', rotation=30)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/departures_by_vessel_class.png", dpi=300)
    plt.savefig("plots/departures_by_vessel_class.pdf")
    print(f"Saved vessel-class departure trends to plots/departures_by_vessel_class.png")
    # plt.show()


def plot_departure_histogram(classified_df):
    # Allowed vessel classes
    allowed_classes = {"Bulk Carrier", "Container Ship", "Tanker"}

    # Filter by allowed vessel types
    df_filtered = classified_df[classified_df["vesselType_clean"].isin(allowed_classes)].copy()

    # Normalize port names and count
    port_counts = df_filtered["nextPort"].str.upper().str.strip().value_counts()
    #port_counts = port_counts.loc[["ROTTERDAM", "LOS ANGELES, CA", "LONG BEACH, CA"]]
    port_counts = port_counts.loc[["ROTTERDAM"]]

    # Plot
    plt.figure(figsize=(8, 5))
    port_counts.plot(kind="bar", color="steelblue", edgecolor="black")

    plt.title("Departures of Selected Vessels by Destination", fontsize=22)
    plt.ylabel("Departures", fontsize=20)
    plt.xlabel("")
    plt.xticks(rotation=15)
    plt.tick_params(axis='both', labelsize=18)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/singapore_departures_filtered.png", dpi=300)
    plt.savefig("plots/singapore_departures_filtered.pdf")
    # plt.show()
    
def map_and_filter_vessel_types(df,
    #output_csv="output/LA_ROT_departures_classified.csv"
    output_csv="output/ROT_departures_classified.csv"
):
    if os.path.exists(output_csv):
        print(f"All departures CSV already exists at '{output_csv}', skipping.")
        return pd.read_csv(output_csv)

    # Mapping raw vessel types to clean labels
    type_mapping = {
        'Bulk Carrier': {'BULK CARRIER', 'BC'},
        'Container Ship': {'CS', 'CONTAINER SHIP'},
        'Tanker': {'TA', 'TANKER', 'CHEMICAL TANKER', 'PETROLEUM/CHEMICAL TANKER', 'CH', 'TAPC'}
    }

    # Flatten to reverse lookup map
    reverse_map = {raw.upper(): clean for clean, raws in type_mapping.items() for raw in raws}

    # Normalize and map vessel types
    df["vesselType_clean"] = df["vesselType"].str.upper().str.strip().map(reverse_map)

    # Filter only rows with recognized vessel types
    df = df.dropna(subset=["vesselType_clean"]).copy()

    print(f"Retained {len(df)} rows with mapped vessel types.")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"All departures saved to '{output_csv}'")
    return df
    
def plot_vessel_type_breakdown(df, destination_ports):
    # Ensure port names are uppercase and stripped
    port_set = set(p.upper() for p in destination_ports)
    df_filtered = df[df["nextPort"].str.upper().str.strip().isin(port_set)]

    # Expected categories from mapping
    allowed_categories = {"Bulk Carrier", "Container Ship", "Tanker"}

    # Filter vessel types to only mapped categories (if any slipped in)
    df_filtered = df_filtered[df_filtered["vesselType_clean"].isin(allowed_categories)]

    # Count and sort
    type_counts = df_filtered["vesselType_clean"].value_counts().sort_values(ascending=False)

    if type_counts.empty:
        print(f"No mapped vessel types for destination(s): {destination_ports}")
        return

    # Plot
    plt.figure(figsize=(7, 5))
    type_counts.plot(kind="bar", color="mediumseagreen", edgecolor="black")
    plt.title(f"Vessel Type Breakdown to\n{' and '.join(destination_ports)}", fontsize=20)
    plt.ylabel("Number of Vessels", fontsize=18)
    plt.xlabel("")
    plt.tick_params(axis='both', labelsize=18)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout(pad=2.0)

    os.makedirs("plots", exist_ok=True)
    label = "_".join(p.replace(",", "").replace(" ", "_").lower() for p in destination_ports)
    plt.savefig(f"plots/vessel_type_breakdown_{label}.png", dpi=300)
    plt.savefig(f"plots/vessel_type_breakdown_{label}.pdf")
    #plt.show()

def plot_tonnage_histograms_by_vessel_class(df, destination_ports):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    pretty_names = {
        "netTonnage": "Net Tonnage",
        "grossTonnage": "Gross Tonnage",
        "deadweight": "Deadweight Tonnage"
    }

    units = {
        "netTonnage": "m$^3$",
        "grossTonnage": "m$^3$",
        "deadweight": "tonnes"
    }

    port_set = set(p.upper() for p in destination_ports)
    df_filtered = df[df["nextPort"].str.upper().str.strip().isin(port_set)].dropna(subset=["vesselType_clean"])

    tonnage_fields = ["grossTonnage", "netTonnage", "deadweight"]
    label = "_".join(p.replace(",", "").replace(" ", "_").lower() for p in destination_ports)
    os.makedirs("plots", exist_ok=True)

    for field in tonnage_fields:
        g = sns.displot(
            data=df_filtered,
            x=field,
            hue="vesselType_clean",
            kind="hist",
            element="step",
            bins=20,
            height=6,
            aspect=1.5,
            linewidth=1.5
        )
        g.set_axis_labels(f"{pretty_names[field]} ({units[field]})", "Count", fontsize=20)
        g.fig.suptitle(f"{pretty_names[field]} by Vessel Class to {' and '.join(destination_ports)}", y=1.03, fontsize=24)
        
        # Tick label sizes
        for ax in g.axes.flat:
            ax.tick_params(axis='both', labelsize=18)
    
        g._legend.set_title("Vessel Class")  # explicitly set legend title
        
        # Legend font size
        if g._legend:
            g._legend.set_title("Vessel Class")
            g._legend.get_title().set_fontsize(20)
            for text in g._legend.texts:
                text.set_fontsize(18)
            g._legend.set_bbox_to_anchor((1.05, 0.5))

        g.savefig(f"plots/tonnage_hist_{field}_{label}.png", dpi=300)
        g.savefig(f"plots/tonnage_hist_{field}_{label}.pdf")
        plt.close(g.fig)
        
def average_rotterdam_departures(
    classified_df,
    size_def_csv="data/vessel_class_definitions.csv",
    output_csv="output/vessel_class_definitions_with_averages.csv"
):
    if not os.path.exists(size_def_csv):
        print(f"File not found: {size_def_csv}")
        return None

    # Load vessel class definitions
    size_df = pd.read_csv(size_def_csv)
    size_df.columns = [c.strip() for c in size_df.columns]
    size_df["Deadweight (tonnes)"] = pd.to_numeric(size_df["Deadweight (tonnes)"], errors="coerce")

    # Prepare classified_df
    classified_df["vesselType_clean"] = classified_df["vesselType_clean"].str.strip()
    classified_df["deadweight"] = pd.to_numeric(classified_df["deadweight"], errors="coerce")
    classified_df["PortDuration"] = pd.to_numeric(classified_df["PortDuration"], errors="coerce")
    classified_df["Average Speed"] = pd.to_numeric(classified_df["Average Speed"], errors="coerce")
    classified_df["vesselSizeCategory"] = None
    classified_df["departure_datetime"] = pd.to_datetime(classified_df["reportedDepartureTime"], errors="coerce")

    # Assign size category using closest deadweight from operational column
    for idx, row in classified_df.iterrows():
        vclass = row["vesselType_clean"]
        dw = row["Capacity - Dwt"]

        if pd.isna(dw):
            continue

        candidates = size_df[size_df["Vessel Class"] == vclass]
        if candidates.empty or candidates["Deadweight (tonnes)"].isna().all():
            continue

        closest = (candidates["Deadweight (tonnes)"] - dw).abs().idxmin()
        classified_df.at[idx, "vesselSizeCategory"] = size_df.at[closest, "Vessel Size"]

    # Determine elapsed time in float years
    valid_times = classified_df["departure_datetime"].dropna()
    if valid_times.empty:
        print("No valid departure datetimes found.")
        return classified_df, size_df

    time_elapsed_seconds = (valid_times.max() - valid_times.min()).total_seconds()
    seconds_per_year = 365.25 * 24 * 3600
    elapsed_years = time_elapsed_seconds / seconds_per_year

    # Filter for Rotterdam departures
    rotterdam_df = classified_df[
        classified_df["nextPort"].str.upper().str.strip() == "ROTTERDAM"
    ].copy()

    # Count total and annualized departures
    total_counts = (
        rotterdam_df
        .groupby(["vesselType_clean", "vesselSizeCategory"])
        .size()
        .reset_index(name="Total Departures")
    )
    total_counts["Annual Rotterdam Departures"] = total_counts["Total Departures"] / elapsed_years
    total_counts["Std Dev (Poisson)"] = np.sqrt(total_counts["Total Departures"]) / elapsed_years

    # Compute average and std of port duration
    avg_duration = (
        rotterdam_df
        .groupby(["vesselType_clean", "vesselSizeCategory"])["PortDuration"]
        .agg(Avg_PortDuration=("mean"), Std_PortDuration=("std"))
        .reset_index()
        .rename(columns={
            "Avg_PortDuration": "Avg Port Duration (days)",
            "Std_PortDuration": "Std Port Duration (days)"
        })
    )

    # Compute average and std of Average Speed
    avg_speed = (
        rotterdam_df
        .groupby(["vesselType_clean", "vesselSizeCategory"])["Average Speed"]
        .agg(Avg_Speed=("mean"), Std_Speed=("std"))
        .reset_index()
        .rename(columns={
            "Avg_Speed": "Avg Speed (knots)",
            "Std_Speed": "Std Speed (knots)"
        })
    )

    # Merge all summaries
    summary = (
        total_counts
        .merge(avg_duration, on=["vesselType_clean", "vesselSizeCategory"], how="left")
        .merge(avg_speed, on=["vesselType_clean", "vesselSizeCategory"], how="left")
    )

    # Merge into size definitions
    size_df = size_df.merge(
        summary,
        left_on=["Vessel Class", "Vessel Size"],
        right_on=["vesselType_clean", "vesselSizeCategory"],
        how="left"
    ).drop(columns=["vesselType_clean", "vesselSizeCategory", "Total Departures"])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    size_df.to_csv(output_csv, index=False)

    print(f"Saved to '{output_csv}' with averages and std devs (Poisson & Speed) over {elapsed_years:.2f} years of data.")
    print(f"Data range: {valid_times.min().date()} to {valid_times.max().date()}")
    
    return classified_df, size_df
    

def enrich_with_operational_data(
    classified_df,
    rot_csv="data/marine_traffic_ais/ROT_departures_operational_data.csv",
    output_csv="output/ROT_departures_classified_enriched.csv"
):
    if not os.path.exists(rot_csv):
        print(f"File not found: {rot_csv}")
        return classified_df

    # Step 1: Filter classified_df for Rotterdam departures
    classified_df = classified_df[
        classified_df["nextPort"].str.upper().str.strip() == "ROTTERDAM"
    ].copy()

    # Step 2: Load operational data
    rot_df = pd.read_csv(rot_csv)
    rot_df["Imo"] = rot_df["Imo"].astype(str)
    classified_df["imoNumber"] = classified_df["imoNumber"].astype(str)

    # Step 3: Compute Voyage Distance per row
    rot_df["Distance Travelled"] = pd.to_numeric(rot_df["Distance Travelled"], errors="coerce")
    rot_df["Distance To Go"] = pd.to_numeric(rot_df["Distance To Go"], errors="coerce")
    rot_df["Voyage Distance"] = rot_df["Distance Travelled"] + rot_df["Distance To Go"]

    # Step 4: Select relevant columns including raw Voyage Distance
    selected_cols = ["Imo", "Average Speed", "Capacity - Dwt", "Built", "Voyage Distance"]
    rot_subset = rot_df[selected_cols]

    # Step 5: Merge directly on ImoNumber
    enriched_df = classified_df.merge(
        rot_subset,
        left_on="imoNumber",
        right_on="Imo",
        how="left"
    ).drop(columns=["Imo"])  # Remove redundant join key

    # Step 6: Save result
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    enriched_df.to_csv(output_csv, index=False)
    print(f"✔ Enriched Rotterdam departures saved to '{output_csv}'")

    return enriched_df


def main():
    all_df = export_all_departures_to_csv()
    filtered_df = filter_departures(all_df)
    plot_departures_per_day(all_df, filtered_df)
    enriched_df = add_vessel_particulars(filtered_df)
    updated_df = update_deadweight_from_csv(enriched_df)
    enriched_updated_df = enrich_with_operational_data(updated_df)
    classified_df = map_and_filter_vessel_types(enriched_updated_df)
    
    plot_departures_by_vessel_class(all_df, classified_df)
#
    plot_departure_histogram(classified_df)
#
    plot_vessel_type_breakdown(classified_df, ["ROTTERDAM"])
#    plot_vessel_type_breakdown(classified_df, ["LOS ANGELES, CA", "LONG BEACH, CA"])
#
    plot_tonnage_histograms_by_vessel_class(classified_df, ["ROTTERDAM"])
#    plot_tonnage_histograms_by_vessel_class(classified_df, ["LOS ANGELES, CA", "LONG BEACH, CA"])
    
    classified_df, updated_definitions = average_rotterdam_departures(classified_df)

if __name__ == "__main__":
    main()



