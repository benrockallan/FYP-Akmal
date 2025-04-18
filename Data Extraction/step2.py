import pandas as pd
import os
import sys
import chardet
import re

# Map from raw column names to standardized ones (except 'Vid' which will be generated)
COLUMN_RENAME_MAP = {
    'cid': 'Cid',
    'repliesToId': 'RepliesToId',
    'text': 'Comments',
    'uniqueId': 'uniqueId',
    'videoWebUrl': 'videoWebUrl'
}

EXPECTED_COLUMNS = ["Vid", "Cid", "RepliesToId", "Comments", "uniqueId", "videoWebUrl"]

def extract_video_id(url):
    if pd.isna(url):
        return None
    match = re.search(r'/video/(\d+)', url)
    return match.group(1) if match else None

def process_csv_file(file_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Detect file encoding
    with open(file_path, 'rb') as f:
        rawdata = f.read()
        detected = chardet.detect(rawdata)
        detected_encoding = detected['encoding']

    encodings_to_try = [detected_encoding, 'utf-8-sig', 'utf-8', 'latin1', 'cp1252', 
                        'iso-8859-1', 'windows-1250', 'windows-1252', 'mac_roman']

    df = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
            break
        except:
            continue

    if df is None:
        raise Exception("Failed to read the CSV file with any of the supported encodings.")

    # Rename columns to match standard format
    df = df.rename(columns={k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns})

    # Extract video ID and assign numeric Vid
    if 'videoWebUrl' in df.columns:
        df['video_id_raw'] = df['videoWebUrl'].apply(extract_video_id)
        unique_vids = {v: i+1 for i, v in enumerate(pd.unique(df['video_id_raw'].dropna()))}
        df['Vid'] = df['video_id_raw'].map(unique_vids)
    else:
        df['Vid'] = None

    # Ensure all expected columns exist
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Keep only expected columns
    df = df[EXPECTED_COLUMNS]

    # Save final standardized file
    output_path = os.path.join("Standardized Comment.csv")
    try:
        df.to_csv(output_path, index=False, encoding=detected_encoding)
    except:
        df.to_csv(output_path, index=False, encoding='utf-8-sig', errors='replace')

    print(f"✅ Processed file saved as Standardized Comment.csv in {output_dir}")

if __name__ == "__main__":
    #if len(sys.argv) != 3:
    #    print("Usage: python script.py <input_csv_file> <output_directory>")
    #    sys.exit(1)

    input_csv = "Testing Comment.csv"
    output_csv = "Standardized Comment.csv"

    process_csv_file(input_csv, output_csv)
