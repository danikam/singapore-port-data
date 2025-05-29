import argparse
import pandas as pd

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Read a CSV and print unique IMO numbers.")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    args = parser.parse_args()

    # Read the CSV file
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Check if the 'imoNumber' column exists
    if 'imoNumber' not in df.columns:
        print("Error: 'imoNumber' column not found in the CSV file.")
        return

    # Get unique IMO numbers
    unique_imos = df['imoNumber'].dropna().unique()
    
    print("Unique IMO Numbers:")
    for imo in unique_imos:
        print(imo)
    print(f"\nTotal number of unique IMO numbers: {len(unique_imos)}")

if __name__ == "__main__":
    main()

