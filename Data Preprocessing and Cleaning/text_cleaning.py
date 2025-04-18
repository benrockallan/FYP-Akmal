#Data Preprocessing And Cleaning
import re
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

#Step 1: Environment Setup
#!pip install pandas numpy openpyxl keybert transformers torch scikit-learn


# Loading the Merged Comment data
comments_df = pd.read_csv("../Data Extraction/Standardized Comments Scaled Down.csv", encoding="utf-8-sig",)

# Loading the Video with Transcription data
videos_df = pd.read_csv("../Data Extraction/Video With Transcription.csv", encoding="utf-8-sig",)

#Inspecting Data
comments_df.head()
videos_df.head()

# Step 2: Handling Missing Values Section
# Check missing values in comments
print(comments_df.isnull().sum())

# Remove missing values or fill with placeholder text
comments_df.dropna(subset=['Comments'], inplace=True)

# Alternatively, to fill:
# comments_df['comments'].fillna("No comment provided", inplace=True)

# Check missing values in videos
print(videos_df.isnull().sum())

videos_df.dropna(subset=['TikTok_Transcription'], inplace=True)

# Step 3: Removing duplicates Section
# Remove duplicates from comments
comments_df.drop_duplicates(subset=['Comments'], inplace=True)

# Remove duplicates from videos
videos_df.drop_duplicates(subset=['TikTok_Transcription'], inplace=True)

# Step4: Text Normalization & Cleaning
def clean_text(text):
    text = str(text).lower()  # convert text to lowercase
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # normalize whitespace
    return text

# Apply cleaning to comments and transcriptions
comments_df['clean_comments'] = comments_df['Comments'].apply(clean_text)
videos_df['clean_transcriptions'] = videos_df['TikTok_Transcription'].apply(clean_text)

#Step 5: Tokenization
# Tokenize the cleaned comments and transcriptions
comments_df['tokens'] = comments_df['clean_comments'].apply(word_tokenize)
videos_df['tokens'] = videos_df['clean_transcriptions'].apply(word_tokenize)

# Step 6: Stopword Removal
stop_words = set(stopwords.words('english'))

# Define function for removing stopwords
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

comments_df['tokens_no_stopwords'] = comments_df['tokens'].apply(remove_stopwords)
videos_df['tokens_no_stopwords'] = videos_df['tokens'].apply(remove_stopwords)

#Step 7: Lemmatization
lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

comments_df['lemmatized'] = comments_df['tokens_no_stopwords'].apply(lemmatize_tokens)
videos_df['lemmatized'] = videos_df['tokens_no_stopwords'].apply(lemmatize_tokens)

#Step 8: Preparing for Keyword Extraction (KeyBERT)
comments_df['final_text'] = comments_df['lemmatized'].apply(lambda x: ' '.join(x))
videos_df['final_text'] = videos_df['lemmatized'].apply(lambda x: ' '.join(x))

#Step 9: Save Preprocessed Data
comments_df.to_csv('Clean Standardized Comment.csv', index=False, encoding="utf-8-sig",)
videos_df.to_csv('Clean Video with Transcription.csv', index=False, encoding="utf-8-sig",)
