import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

# Load the data
df = pd.read_csv(r"C:\Users\ViezPC1\Pythonprojects\sentiment_aynalysis\glassdoor_reviews.csv")

# Preprocessing function
def preprocess_text(text):
    if isinstance(text, str):  # Check if the text is a string
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
    else:
        text = ''  # If it's not a string, return an empty string or handle accordingly
    return text

# Apply preprocessing to 'pros' and 'cons' columns
df['cleaned_pros'] = df['pros'].apply(preprocess_text)
df['cleaned_cons'] = df['cons'].apply(preprocess_text)
print(df[['pros', 'cleaned_pros', 'cons', 'cleaned_cons']].head())
