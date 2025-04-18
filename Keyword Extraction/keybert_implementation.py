import pandas as pd
from keybert import KeyBERT
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import threading
import time
import os
import emoji  # Import the emoji library

# Download NLTK resources (Consider doing this once outside the main script execution if possible)
# Ensure NLTK data path is correctly set up if needed
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt', quiet=True)
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     nltk.download('stopwords', quiet=True)
# try:
#     nltk.data.find('corpora/wordnet')
# except nltk.downloader.DownloadError:
#     nltk.download('wordnet', quiet=True)
# try:
#     nltk.data.find('corpora/omw-1.4')
# except nltk.downloader.DownloadError:
#     nltk.download('omw-1.4', quiet=True)


class KeywordExtractionApp:
    def __init__(self):
        # Initialize variables
        self.comments_file_path = ""
        self.videos_file_path = ""
        self.output_dir = os.getcwd()
        self.progress_var = 0
        self.status_var = "Ready"
        self.processing_thread = None
        self.stop_requested = False
        self.log_text = []  # Use a list to store log messages
        self.kw_model = KeyBERT() # Initialize KeyBERT here

    def log(self, message):
        """
        Adds a message to the log.  Now stores in a list.
        """
        self.log_text.append(message)

    def update_status(self, message):
        """
        Updates the status variable.
        """
        self.status_var = message

    def update_progress(self, value):
        """
        Updates the progress variable.
        """
        self.progress_var = value

    def start_processing(self, comments_file, videos_file, output_directory):
        """
        Starts the keyword extraction process in a separate thread.
        """
        if not comments_file and not videos_file:
            self.log("Error: Please select at least one input file.")
            self.update_status("Error: No input file selected")
            return

        self.comments_file_path = comments_file
        self.videos_file_path = videos_file
        self.output_dir = output_directory
        self.stop_requested = False
        self.log_text = []  # Clear log
        self.update_progress(0)
        self.update_status("Starting...")

        self.processing_thread = threading.Thread(target=self.process_files)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def stop_processing(self):
        """
        Sets the stop_requested flag to True.
        """
        self.stop_requested = True
        self.log("Stop requested. Waiting for current batch to complete...")
        self.update_status("Stopping...")

    def process_files(self):
        """
        Main function to process the files and extract keywords.  This is run in a thread.
        """
        try:
            self.log("Starting keyword extraction process...")
            self.update_status("Initializing...")

            comment_column = "Comments"  # Hardcoded
            transcription_column = "TikTok_Transcription"  # Hardcoded - changed to TikTok_Transcription
            top_n = 5
            batch_size = 100
            output_dir = self.output_dir

            # --- Text cleaning and processing functions ---
            self.log("Setting up text processing functions...")

            def clean_text(text):
                """Cleans the text, handling potential errors."""
                if not isinstance(text, str):
                    return ""
                text = text.lower()
                text = re.sub(r'http\S+', '', text)
                text = re.sub(r'\d+', '', text)
                emoji_chars = "".join(emoji.EMOJI_DATA.keys())
                keep_pattern = f"[^\\w\\s{re.escape(emoji_chars)}]"  # Corrected raw string
                text = re.sub(keep_pattern, '', text, flags=re.UNICODE)
                text = re.sub(r'\s+', ' ', text).strip()
                return text

            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()

            def process_text(text):
                """Processes the text, handling potential errors."""
                try:
                    tokens = word_tokenize(text)
                    tokens_no_stopwords = [
                        word for word in tokens if word not in stop_words
                    ]
                    lemmatized = [lemmatizer.lemmatize(token) for token in tokens_no_stopwords]
                    return ' '.join(lemmatized)
                except Exception as e:
                    self.log(f"Error processing text: {e}")
                    return ""

            # --- Load data (Ensure error handling for file loading) ---
            comments_df = None
            videos_df = None

            try:
                if self.comments_file_path:
                    self.log(f"Loading comments data from {self.comments_file_path}...")
                    self.update_status("Loading comments data...")
                    if self.comments_file_path.lower().endswith('.csv'):
                        comments_df = pd.read_csv(self.comments_file_path)
                    else:
                        comments_df = pd.read_excel(self.comments_file_path)
                    self.log(f"Successfully loaded comments data with {len(comments_df)} rows")
                    self.log(f"Columns: {comments_df.columns.tolist()}")
                    # Validate comment column
                    if comment_column not in comments_df.columns:
                        self.log(
                            f"Warning: '{comment_column}' not found. Available: {comments_df.columns.tolist()}. Attempting fallback..."
                        )
                        potential_columns = [col for col in comments_df.columns if 'comment' in col.lower()]
                        if potential_columns:
                            comment_column = potential_columns[0]
                            self.log(f"Using '{comment_column}' instead.")
                        else:
                            raise ValueError(f"Specified comment column '{self.comment_column_var}' not found.")

            except Exception as e:
                self.log(f"Error loading comments file: {e}")
                self.update_status("Error loading comments file")
                comments_df = None  # Ensure it's None if loading fails

            try:
                if self.videos_file_path:
                    self.log(f"Loading videos data from {self.videos_file_path}...")
                    self.update_status("Loading videos data...")
                    if self.videos_file_path.lower().endswith('.csv'):
                        videos_df = pd.read_csv(self.videos_file_path)
                    else:
                        videos_df = pd.read_excel(self.videos_file_path)
                    self.log(f"Successfully loaded videos data with {len(videos_df)} rows")
                    self.log(f"Columns: {videos_df.columns.tolist()}")
                    # Validate transcription column
                    if transcription_column not in videos_df.columns:
                        self.log(
                            f"Warning: '{transcription_column}' not found. Available: {videos_df.columns.tolist()}. Attempting fallback..."
                        )
                        potential_columns = [
                            col for col in videos_df.columns if 'transcript' in col.lower() or 'text' in col.lower()
                        ]
                        if potential_columns:
                            transcription_column = potential_columns[0]
                            self.log(f"Using '{transcription_column}' instead.")
                        else:
                            raise ValueError(
                                f"Specified transcription column '{self.transcription_column_var}' not found."
                        )

            except Exception as e:
                self.log(f"Error loading videos file: {e}")
                self.update_status("Error loading videos file")
                videos_df = None  # Ensure it's None if loading fails

            # --- Initialize KeyBERT ---
            self.log("Initializing KeyBERT model...")
            self.update_status("Initializing KeyBERT model...")
            #kw_model = KeyBERT()  # Initialize KeyBERT here
            kw_model = self.kw_model

            # --- Function to extract keywords ---
            def extract_keywords(text, top_n_kw=5):
                """Extracts keywords, handling errors and short texts."""
                if not text or len(text.split()) < 3:
                    return []
                try:
                    keywords = kw_model.extract_keywords(
                        text,
                        keyphrase_ngram_range=(1, 2),
                        stop_words='english',  # KeyBERT handles its own stopwords
                        use_mmr=True,
                        diversity=0.7,
                        top_n=top_n_kw,
                    )
                    return keywords if keywords else []
                except Exception as e:
                    self.log(f"Error extracting keywords for text chunk: {e}")
                    return []

            # --- Process comments if available ---
            if comments_df is not None:
                self.log("\nProcessing comments data...")
                self.update_status("Processing comments data...")

                # Clean and preprocess text
                self.log("Cleaning comments...")
                comments_df['clean_text'] = comments_df[comment_column].apply(clean_text)
                self.log("Preprocessing comments (tokenize, stopwords, lemmatize)...")
                comments_df['processed_text'] = comments_df['clean_text'].apply(process_text)

                # Extract keywords in batches
                self.log("Extracting keywords from comments...")
                keywords_list_comments = []
                total_comments = len(comments_df)

                for i in range(0, total_comments, batch_size):
                    if self.stop_requested:
                        self.log("Comment processing stopped by user.")
                        break

                    batch_end = min(i + batch_size, total_comments)
                    current_batch_num = (i // batch_size) + 1
                    total_batches = (total_comments + batch_size - 1) // batch_size

                    self.log(f"Processing comment batch {current_batch_num}/{total_batches} (rows {i + 1} to {batch_end})")
                    self.update_status(f"Processing comments: {batch_end}/{total_comments}")

                    batch_texts = comments_df['processed_text'].iloc[i:batch_end]
                    batch_keywords = [extract_keywords(text, top_n) for text in batch_texts] # Pass kw_model
                    keywords_list_comments.extend(batch_keywords)

                    progress = (batch_end / total_comments) * 50  # Comments take first 50%
                    self.update_progress(progress)

                if not self.stop_requested:
                    comments_df['keywords'] = keywords_list_comments
                    comments_df['keywords_only'] = comments_df['keywords'].apply(
                        lambda kw_list: [k for k, _ in kw_list] if isinstance(kw_list, list) else []
                    )

                    output_path = os.path.join(output_dir, 'comments_with_keywords.csv')
                    try:
                        self.log(f"Saving processed comments to {output_path}")
                        comments_df.to_csv(output_path, index=False, encoding='utf-8-sig')  # Ensure UTF-8
                        self.log("Saved comments successfully.")
                        self.log("\nSample keywords from comments:")
                        for idx in range(min(3, len(comments_df))):
                            original_comment = comments_df[comment_column].iloc[idx]
                            extracted_kws = comments_df['keywords'].iloc[idx]
                            self.log(f"Comment (Original): {str(original_comment)[:100]}...")
                            self.log(f"Keywords: {extracted_kws}")
                            self.log("-" * 20)
                    except Exception as e:
                        self.log(f"Error saving comments file: {e}")
                        self.update_status("Error saving comments file")

            # --- Process videos if available ---
            if videos_df is not None and not self.stop_requested:
                self.log("\nProcessing video transcription data...")
                self.update_status("Processing video transcriptions...")

                self.log("Cleaning transcriptions...")
                videos_df['clean_text'] = videos_df[transcription_column].apply(clean_text)
                self.log("Preprocessing transcriptions (tokenize, stopwords, lemmatize)...")
                videos_df['processed_text'] = videos_df['clean_text'].apply(process_text)

                self.log("Extracting keywords from transcriptions...")
                keywords_list_videos = []
                total_videos = len(videos_df)

                for i in range(0, total_videos, batch_size):
                    if self.stop_requested:
                        self.log("Video processing stopped by user.")
                        break

                    batch_end = min(i + batch_size, total_videos)
                    current_batch_num = (i // batch_size) + 1
                    total_batches = (total_videos + batch_size - 1) // batch_size

                    self.log(f"Processing transcription batch {current_batch_num}/{total_batches} (rows {i + 1} to {batch_end})")
                    self.update_status(f"Processing transcriptions: {batch_end}/{total_videos}")

                    batch_texts = videos_df['processed_text'].iloc[i:batch_end]
                    batch_keywords = [extract_keywords(text, top_n) for text in batch_texts] # Pass kw_model
                    keywords_list_videos.extend(batch_keywords)

                    # Update progress (videos take second 50% or full 100% if no comments)
                    base_progress = 50 if comments_df is not None else 0
                    progress = base_progress + (batch_end / total_videos) * (100 - base_progress)
                    self.update_progress(progress)

                if not self.stop_requested:
                    videos_df['keywords'] = keywords_list_videos
                    videos_df['keywords_only'] = videos_df['keywords'].apply(
                        lambda kw_list: [k for k, _ in kw_list] if isinstance(kw_list, list) else []
                    )

                    output_path = os.path.join(output_dir, 'videos_with_keywords.csv')
                    try:
                        self.log(f"Saving processed videos to {output_path}")
                        videos_df.to_csv(output_path, index=False, encoding='utf-8-sig')  # Ensure UTF-8
                        self.log("Saved videos successfully.")
                        self.log("\nSample keywords from videos:")
                        for idx in range(min(3, len(videos_df))):
                            original_transcript = videos_df[transcription_column].iloc[idx]
                            extracted_kws = videos_df['keywords'].iloc[idx]
                            self.log(f"Transcription (Original): {str(original_transcript)[:100]}...")
                            self.log(f"Keywords: {extracted_kws}")
                            self.log("-" * 20)
                    except Exception as e:
                        self.log(f"Error saving videos file: {e}")
                        self.update_status("Error saving videos file")

            # --- Completion message ---
            if self.stop_requested:
                self.update_status("Processing stopped by user")
                self.log("Processing stopped before completion.")
            else:
                self.update_status("Processing completed")
                self.log("Keyword extraction completed successfully!")
                self.update_progress(100)

        except FileNotFoundError as e:
            self.log(f"Error: Input file not found - {e}")
            self.update_status("Error: File not found")
        except ValueError as e:  # Catch column errors
            self.log(f"Configuration Error: {e}")
            self.update_status("Error: Column not found")
        except Exception as e:
            self.log(f"An unexpected error occurred during processing: {e}")
            self.update_status("Error occurred")
        finally:
            # Ensure buttons are reset in the main thread
            self.processing_thread = None #clear thread
            self.update_status("Ready")

    def get_progress(self):
        """
        Returns the current progress value.
        """
        return self.progress_var

    def get_status(self):
        """
        Returns the current status message.
        """
        return self.status_var

    def get_log(self):
        """
        Returns the current log messages.
        """
        return self.log_text


if __name__ == "__main__":
    import time
    import os
    import emoji  # Import emoji

    # Download nltk resources
    # try:
    #     nltk.data.find('tokenizers/punkt')
    # except nltk.downloader.DownloadError:
    #     nltk.download('punkt', quiet=True)
    # try:
    #     nltk.data.find('corpora/stopwords')
    # except nltk.downloader.DownloadError:
    #     nltk.download('stopwords', quiet=True)
    # try:
    #     nltk.data.find('corpora/wordnet')
    # except nltk.downloader.DownloadError:
    #     nltk.download('wordnet', quiet=True)
    # try:
    #     nltk.data.find('corpora/omw-1.4')
    # except nltk.downloader.DownloadError:
    #     nltk.download('omw-1.4', quiet=True)

    # Use the specified file paths directly
    comments_file_path = r"/Users/bakri/Development/fyp_supervision/FYP-Akmal/Data Preprocessing and Cleaning/Clean Standardized Comment.csv"
    videos_file_path = r"/Users/bakri/Development/fyp_supervision/FYP-Akmal/Data Preprocessing and Cleaning/Clean Video with Transcription.csv"
    output_dir = r"/Users/bakri/Development/fyp_supervision/FYP-Akmal/Keyword Extraction"  # Corrected

    app = KeywordExtractionApp()  # Create instance of the class

    # Start processing
    app.start_processing(comments_file_path, videos_file_path, output_dir)

    # Keep the script running until processing is complete or stopped
    while app.processing_thread and app.processing_thread.is_alive():
        time.sleep(1)  # Check every second
        progress = app.get_progress()
        status = app.get_status()
        log_messages = app.get_log()  # Get log messages

        # Print progress, status, and log messages
        print(f"\nProgress: {progress:.2f}%")
        print(f"Status: {status}")
        print("Log:")
        for message in log_messages:  # Print all messages
            print(f"  {message}")

    # Print final progress, status and log
    progress = app.get_progress()
    status = app.get_status()
    log_messages = app.get_log()
    print(f"\nFinal Progress: {progress:.2f}%")
    print(f"Final Status: {status}")
    print("Final Log:")
    for message in log_messages:
        print(f"  {message}")
