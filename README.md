# Singapore Port Vessel Departure Analysis

This code analyzes vessel departures from Singapore using data from the MPA (Maritime and Port Authority of Singapore). It includes tools to fetch raw API data, enrich it with vessel particulars, and generate visualizations highlighting patterns in departures to the following destination ports: Rotterdam, Los Angeles, and Long Beach.

# Setup & Usage

## Install dependencies

Make sure you have the following installed:
* `jq` (for JSON parsing): `brew install jq`
* `coreutils` (for gdate): `brew install coreutils`
* Python packages:

```bash
pip install pandas matplotlib seaborn requests
```

## Create API key

Register for a free account with the [Singapore Maritime Data Hub](https://sg-mdh.mpa.gov.sg/). 

Once you have an account, create an API Key for the "Vessel Departure Declaration", "Vessels Departed", "Vessels Due To Depart", and "Vessel Particulars" API products (along with any others of interest). This will include a "Consumer Key" that is used for subsequent data fetching.

## Fetch data from the API

Use the provided shell script to collect arrival or departure data.

```bash
bash source/get_arrival_departure_data.sh <API_KEY> departure
```

* Replace `<API_KEY>` with the Consumer Key generated above.
* This will save a combined JSON of results in `data/vessel_departures.json`.

## Run analysis and generate plots

```bash
python source/analyze_departure_data.py --api_key <API_KEY>
```

This will:

* Export and deduplicate the raw JSON to `output/all_departures.csv`
* Filter for Rotterdam, LA, and Long Beach routes
* Enrich the dataset with vessel particulars via API
* Apply manual deadweight overrides if available
* Generate plots in the `plots/` folder
