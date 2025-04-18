#Make the Datasets Smaller

import pandas as pd

def scale_down_comments(df, min_comments=5, min_replies=5, max_rows_per_vid=5):
    """
    Scales down the number of comments and replies for each Vid in a DataFrame,
    and limits the number of rows per Vid.  Handles emojis correctly by using 'utf-8-sig' encoding.

    Args:
        df (pd.DataFrame): The input DataFrame containing comment data, with columns
            'Vid', 'Cid', 'RepliesToId', 'Comments', 'uniqueId', and 'videoWebUrl'.
        min_comments (int, optional): The minimum number of comments to retain for each Vid.
            Defaults to 5.
        min_replies (int, optional): The minimum number of replies to retain for each Vid.
            Defaults to 5.
        max_rows_per_vid (int, optional): The maximum number of rows to keep for each Vid.
            Defaults to 5.

    Returns:
        pd.DataFrame: A new DataFrame with the scaled-down comments and replies,
        with a maximum of max_rows_per_vid rows per Vid. Returns an empty DataFrame
        if the input is invalid.
    """
    # Check if the input DataFrame is valid
    if not isinstance(df, pd.DataFrame):
        print("Error: Input must be a pandas DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Check for the existence of required columns
    required_columns = ['Vid', 'Cid', 'RepliesToId', 'Comments', 'uniqueId', 'videoWebUrl']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: DataFrame must contain columns: {required_columns}")
        return pd.DataFrame()  # Return an empty DataFrame

    # Check if the DataFrame is empty
    if df.empty:
        print("Warning: Input DataFrame is empty. Returning an empty DataFrame.")
        return pd.DataFrame()

    # Create a copy to avoid modifying the original DataFrame in place.
    df_scaled = df.copy()

    # Group by 'Vid'
    grouped = df_scaled.groupby('Vid')

    # Function to scale down comments and replies for a single Vid
    def scale_vid_comments(group):
        num_comments = len(group)
        num_replies = group['RepliesToId'].count()  # Count non-NaN replies

        if num_comments > min_comments or num_replies > min_replies:
            # Calculate the number of comments and replies to keep
            n_comments_to_keep = max(min_comments, int(num_comments))
            n_replies_to_keep = max(min_replies, int(num_replies))

            # Filter non-null replies
            valid_replies = group[group['RepliesToId'].notna()]

            # Sample comments and replies
            sampled_comments = group.sample(
                n=min(n_comments_to_keep, num_comments), replace=False
            )  # Fix: ensure sample size is not larger than the group size.

            if not valid_replies.empty:
                sampled_replies = valid_replies.sample(
                    n=min(n_replies_to_keep, valid_replies.shape[0]), replace=False
                )  # Fix: ensure sample size is not larger than the valid_replies size.
                # Combine sampled comments and replies.  Prioritize keeping all *replies*
                combined_sampled = pd.concat(
                    [
                        sampled_replies,
                        sampled_comments[
                            ~sampled_comments.index.isin(sampled_replies.index)
                        ],
                    ]
                )
                # Limit to max_rows_per_vid
                return combined_sampled.head(max_rows_per_vid)
            else:
                # Limit to max_rows_per_vid
                return sampled_comments.head(max_rows_per_vid)
        else:
            # Limit to max_rows_per_vid
            return group.head(max_rows_per_vid)  # Return the original group if no scaling is needed

    # Apply the scaling function to each group
    df_scaled = grouped.apply(scale_vid_comments).reset_index(level=0, drop=True)

    return df_scaled

# Load the CSV file into a DataFrame, using 'utf-8-sig' encoding
df = pd.read_csv('Standardized Comment.csv', encoding="utf-8-sig")

# Scale down the comments and replies
df_scaled = scale_down_comments(df)

# Print the first few rows of the scaled DataFrame to verify the result
if not df_scaled.empty:
    print("Scaled Down Data:")
    print(df_scaled.head())
    # Save the scaled DataFrame to a new CSV file, using utf-8-sig
    df_scaled.to_csv('Standardized Comments Scaled Down.csv', index=False, encoding="utf-8-sig")
else:
    print("No data to display or save.")
