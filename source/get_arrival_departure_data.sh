#!/bin/bash

# ==== CONFIGURATION ====
API_KEY="$1"
if [ -z "$API_KEY" ]; then
    echo "Usage: $0 <API_KEY> [arrival|departure]"
    exit 1
fi

START_DATE="2023-01-01 00:00:00"
END_DATE="2025-05-07 23:59:59"
HOURS_STEP=2000
MAX_CHUNKS=100
OUTPUT_DIR="data"
# =======================

# Check required tools
if ! command -v gdate &> /dev/null; then
    echo "Error: 'gdate' not found. Run: brew install coreutils"
    exit 1
fi
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' not found. Run: brew install jq"
    exit 1
fi

# Fetch function for arrivals/departures
fetch_data() {
    local TYPE=$1
    local BASE_URL="https://sg-mdh-api.mpa.gov.sg/v1/vessel/${TYPE}declaration/pastNhours"
    local OUTPUT_FILE="${OUTPUT_DIR}/vessel_${TYPE}s.json"
    local TMP_DIR
    rm ${OUTPUT_FILE}
    TMP_DIR=$(mktemp -d)
    mkdir -p "$OUTPUT_DIR"

    echo "Fetching vessel $TYPE data..."

    START_EPOCH=$(gdate -u -d "$START_DATE" +%s)
    END_EPOCH=$(gdate -u -d "$END_DATE" +%s)
    CURRENT_EPOCH=$((START_EPOCH + HOURS_STEP * 3600))
    COUNTER=0

    LAST_QUERY_EPOCH=0
    while [ "$CURRENT_EPOCH" -le "$END_EPOCH" ]; do
        HOURS_THIS_CHUNK=$HOURS_STEP
        QUERY_EPOCH=$((CURRENT_EPOCH))
        CURRENT_DATETIME=$(gdate -u -d "@$CURRENT_EPOCH" "+%Y-%m-%d%%20%H:%M:%S")
        CHUNK_FILE="$TMP_DIR/part_$COUNTER.json"

        echo "Fetching chunk $COUNTER: $CURRENT_DATETIME for $HOURS_THIS_CHUNK hours"
        CHUNK_URL="$BASE_URL?datetime=$CURRENT_DATETIME&hours=$HOURS_THIS_CHUNK"
        #echo "Running curl: curl -s -X GET --header \"apikey: $API_KEY\" \"$CHUNK_URL\" -o \"$CHUNK_FILE\""
        curl -s -X GET --header "apikey: $API_KEY" \
            "$CHUNK_URL" \
            -o "$CHUNK_FILE"

        LAST_QUERY_EPOCH=$CURRENT_EPOCH
        
        if jq -e '. | length > 0' "$CHUNK_FILE" > /dev/null; then
            # Keep entire chunk intact
            cp "$CHUNK_FILE" "$TMP_DIR/part_flat_$COUNTER.json"
        else
            echo "Chunk $COUNTER is empty, skipping."
            rm "$CHUNK_FILE"
        fi

        ((COUNTER++))
        CURRENT_EPOCH=$((CURRENT_EPOCH + HOURS_STEP * 3600))
        if [ "$COUNTER" -ge "$MAX_CHUNKS" ]; then
            echo "Reached max chunk limit ($MAX_CHUNKS), exiting."
            break
        fi
    done
    
    if [ "$QUERY_EPOCH" -lt "$END_EPOCH" ]; then
        echo "Preparing final catch-up request from $(gdate -u -d "@$QUERY_EPOCH") to $END_DATE"
        FINAL_HOURS=$(( (END_EPOCH - QUERY_EPOCH) / 3600 ))
        if [ "$FINAL_HOURS" -gt 0 ]; then
            FINAL_EPOCH=$((QUERY_EPOCH + FINAL_HOURS * 3600))
            FINAL_DATETIME=$(gdate -u -d "@$FINAL_EPOCH" "+%Y-%m-%d%%20%H:%M:%S")
            FINAL_FILE="$TMP_DIR/part_flat_final.json"
            FINAL_URL="$BASE_URL?datetime=$FINAL_DATETIME&hours=$FINAL_HOURS"

            echo "→ Final chunk datetime: $FINAL_DATETIME"
            echo "→ Final hours         : $FINAL_HOURS"
            #echo "→ Final URL           : $FINAL_URL"
            #echo "Running curl: curl -s -X GET --header \"apikey: $API_KEY\" \"$FINAL_URL\" -o \"$FINAL_FILE\""

            curl -s -X GET --header "apikey: $API_KEY" \
                "$FINAL_URL" \
                -o "$FINAL_FILE"

            if jq -e 'type == "array" and length > 0' "$FINAL_FILE" > /dev/null 2>&1; then
                echo "✔ Final chunk has data"
            else
                echo "⚠ Final chunk is not a valid array or is empty. Skipping."
                rm -f "$FINAL_FILE"
            fi
        fi
    fi

    # Merge all JSON arrays into one array of objects
    jq -s '[.[][]]' "$TMP_DIR"/part_flat_*.json > "$OUTPUT_FILE"

    echo "Final output saved to $OUTPUT_FILE"
    rm -r "$TMP_DIR"
}

# === Main Control ===
case "$1" in
    arrival)
        fetch_data "arrival"
        ;;
    departure)
        fetch_data "departure"
        ;;
    *)
        fetch_data "arrival"
        fetch_data "departure"
        ;;
esac
