import pandas as pd
import re
import requests
from tqdm import tqdm

# Define the paths to your CSV files
input_csv = "Testing Video.csv"
output_csv = "Video with Transcription.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(input_csv)

# Display the original DataFrame (optional)
print("Original DataFrame:")
print(df.head())

# Define columns to always keep
author_columns = [
    "authorMeta/name",
    "authorMeta/nickName",
    "authorMeta/profileUrl",
    "webVideoUrl"
]

# Rename columns for clarity
df.rename(columns={
    "authorMeta/name": "Name",
    "authorMeta/nickName": "NickName",
    "authorMeta/profileUrl": "ProfileUrl",
    "webVideoUrl": "TikTok_Link"
}, inplace=True)

# Identify subtitle columns dynamically
downloadlink_cols = [c for c in df.columns if re.match(r"^videoMeta/subtitleLinks/\d+/downloadLink$", c)]
language_cols = [c for c in df.columns if re.match(r"^videoMeta/subtitleLinks/\d+/language$", c)]

# Map indices to column names
downloadlink_map = {re.findall(r"(\d+)", c)[0]: c for c in downloadlink_cols}
language_map = {re.findall(r"(\d+)", c)[0]: c for c in language_cols}

# Collect subtitle links with eng-US and create separate columns
def extract_eng_us_links(row):
    result = {}
    for idx, lang_col in language_map.items():
        if idx in downloadlink_map and row[lang_col] == "eng-US":
            result[f"eng_us_subtitle_link_{idx}"] = row[downloadlink_map[idx]]
    return pd.Series(result)

# Apply extraction function and combine results
df_eng_us_links = df.apply(extract_eng_us_links, axis=1)
df_combined = pd.concat([df[["Name", "NickName", "ProfileUrl", "TikTok_Link"]], df_eng_us_links], axis=1)

# Remove rows without any eng-US subtitle links
df_filtered = df_combined.dropna(how='all', subset=df_eng_us_links.columns)

# Function to fetch and return subtitle text
def fetch_subtitle_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Failed to fetch subtitle from {url}: {e}")
        return None

# Fetch subtitle transcription from the first available eng-US subtitle link
def get_first_subtitle_transcription(row):
    for col in sorted(df_eng_us_links.columns):
        if pd.notnull(row[col]):
            return fetch_subtitle_text(row[col])
    return None

# Add subtitle transcription column with progress bar
tqdm.pandas(desc="Fetching subtitles")
df_filtered["TikTok_Transcription"] = df_filtered.progress_apply(get_first_subtitle_transcription, axis=1)

# Drop rows that do not contain "WEBVTT" in the TikTok_Transcription
df_filtered = df_filtered[df_filtered["TikTok_Transcription"].str.contains("WEBVTT", na=False)]

# Drop all subtitle link columns after transcription is added
df_filtered.drop(columns=df_eng_us_links.columns, inplace=True)

# Add a new column 'Vid' as primary key
df_filtered.reset_index(drop=True, inplace=True)
df_filtered.insert(0, 'Vid', range(1, len(df_filtered) + 1))

# Display the filtered DataFrame with transcriptions (optional)
print("\nFiltered DataFrame with TikTok Transcriptions and Vid:")
print(df_filtered.head())

# Write the resulting DataFrame to a new CSV file
df_filtered.to_csv(output_csv, index=False)
print(f"\nFiltered data with transcriptions has been saved to {output_csv}")
