import time
import string
import nltk
import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import matplotlib.pyplot as plt

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        text = ' '.join([word for word in text.split() if word not in stop_words])
    else:
        text = ''
    return text

# Function to load and preprocess data
def load_and_preprocess_data(filepath):
    try:
        df = pd.read_csv(filepath, quotechar='"', escapechar='\\', on_bad_lines='skip')
        df['review_text'] = df['pros'] + ' ' + df['cons']
        df['cleaned_review_text'] = df['review_text'].apply(preprocess_text)
        df = df.dropna(subset=['overall_rating'])
        return df
    except Exception as e:
        print(f"Error loading or preprocessing data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

# Load and preprocess data
df = load_and_preprocess_data(r"C:\Users\ViezPC1\Pythonprojects\sentiment_aynalysis\glassdoor_reviews.csv")

# Group data by firm and aggregate
def group_data(df):
    grouped = df.groupby('firm').agg({
        'overall_rating': 'mean',
        'pros': 'count',
        'cons': 'count'
    }).reset_index()

    grouped = grouped.rename(columns={'pros': 'num_pros', 'cons': 'num_cons'})
    grouped['num_reviews'] = grouped['num_pros'] + grouped['num_cons']
    
    return grouped

# Train and save models
def train_models(X_train_tfidf, y_train):
    model_lr = LinearRegression()
    model_rf = RandomForestRegressor(random_state=42)
    model_gb = GradientBoostingRegressor(random_state=42)

    model_lr.fit(X_train_tfidf, y_train)
    model_rf.fit(X_train_tfidf, y_train)
    model_gb.fit(X_train_tfidf, y_train)

    dump(model_lr, 'linear_regression_model.joblib')
    dump(model_rf, 'random_forest_model.joblib')
    dump(model_gb, 'gradient_boosting_model.joblib')

    return model_lr, model_rf, model_gb

# Streamlit UI
st.title('Glassdoor Reviews Analysis')

# Group by firm and aggregate data
grouped_df = group_data(df)

# Display data overview
st.write('### Data Overview')
st.write(df.head())

# Display grouped data by firm
st.write('### Grouped Data by Firm')
st.write(grouped_df.head())

# Search Bar for Firms
search_query = st.text_input("Search for a company:")
if search_query:
    search_query = search_query.strip().lower()
    time.sleep(0.5)  # Debouncing
    search_results = grouped_df[grouped_df['firm'].str.contains(search_query, na=False)]
    if not search_results.empty:
        st.write(f"### Search Results for '{search_query}'")
        st.write(search_results)
    else:
        st.write(f"No companies found matching '{search_query}'")

# Filters for overall rating and number of reviews
min_rating, max_rating = st.sidebar.slider(
    'Select rating range',
    min_value=float(grouped_df['overall_rating'].min()),
    max_value=float(grouped_df['overall_rating'].max()),
    value=(float(grouped_df['overall_rating'].min()), float(grouped_df['overall_rating'].max()))
)

min_reviews, max_reviews = st.sidebar.slider(
    'Select number of reviews range',
    min_value=int(grouped_df['num_reviews'].min()),
    max_value=int(grouped_df['num_reviews'].max()),
    value=(int(grouped_df['num_reviews'].min()), int(grouped_df['num_reviews'].max()))
)

min_pros, max_pros = st.sidebar.slider(
    'Select number of pros range',
    min_value=int(grouped_df['num_pros'].min()),
    max_value=int(grouped_df['num_pros'].max()),
    value=(int(grouped_df['num_pros'].min()), int(grouped_df['num_pros'].max()))
)

min_cons, max_cons = st.sidebar.slider(
    'Select number of cons range',
    min_value=int(grouped_df['num_cons'].min()),
    max_value=int(grouped_df['num_cons'].max()),
    value=(int(grouped_df['num_cons'].min()), int(grouped_df['num_cons'].max()))
)

# Apply filters
filtered_df = grouped_df[
    (grouped_df['overall_rating'] >= min_rating) & 
    (grouped_df['overall_rating'] <= max_rating) & 
    (grouped_df['num_reviews'] >= min_reviews) & 
    (grouped_df['num_pros'] >= min_pros) & 
    (grouped_df['num_cons'] <= max_cons)
]
filtered_df.index = filtered_df.index + 1
filtered_df.index.name = '#'

# Display filtered data
st.write(f"### Filtered Data")
st.write(filtered_df[['firm', 'overall_rating', 'num_reviews', 'num_pros', 'num_cons']])

# Slider to select the top N firms to display
top_n = st.slider('Select number of top firms to display:', min_value=1, max_value=50, value=10)

# Display top N firms by average rating
top_firms = filtered_df.nlargest(top_n, 'overall_rating')
st.write(f'### Top {top_n} Firms by Average Rating')
st.bar_chart(top_firms.set_index('firm')['overall_rating'])

# Display top N firms by number of reviews
top_firms_reviews = filtered_df.nlargest(top_n, 'num_reviews')
st.write(f'### Top {top_n} Firms by Number of Reviews')
st.bar_chart(top_firms_reviews.set_index('firm')['num_reviews'])

# Word Cloud for pros and cons
# pros_text = ' '.join(df['pros'].dropna())
# cons_text = ' '.join(df['cons'].dropna())

# pros_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(pros_text)
# cons_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cons_text)

# st.write("### Word Cloud for Pros")
# st.image(pros_wordcloud.to_array())

# st.write("### Word Cloud for Cons")
# st.image(cons_wordcloud.to_array())

# Combine 'pros' and 'cons' into a single text column for modeling
df['review_text'] = df['pros'] + ' ' + df['cons']
df['cleaned_review_text'] = df['review_text'].apply(preprocess_text)

# Drop rows with missing values in the target column
df = df.dropna(subset=['overall_rating'])

# Features and target
X = df['cleaned_review_text']
y = df['overall_rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Check if models exist; if not, train them
try:
    model_lr = load('linear_regression_model.joblib')
    model_rf = load('random_forest_model.joblib')
    model_gb = load('gradient_boosting_model.joblib')
except FileNotFoundError:
    model_lr, model_rf, model_gb = train_models(X_train_tfidf, y_train)

# Predict on the test set for models
y_pred_lr = model_lr.predict(X_test_tfidf)
y_pred_rf = model_rf.predict(X_test_tfidf)
y_pred_gb = model_gb.predict(X_test_tfidf)

# Evaluate the models
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_gb = mean_squared_error(y_test, y_pred_gb)

r2_lr = model_lr.score(X_test_tfidf, y_test)
r2_rf = model_rf.score(X_test_tfidf, y_test)
r2_gb = model_gb.score(X_test_tfidf, y_test)

# Display model performance
st.write(f'Linear Regression - Mean Squared Error: {mse_lr:.4f}, R-squared: {r2_lr:.4f}')
st.write(f'Random Forest - Mean Squared Error: {mse_rf:.4f}, R-squared: {r2_rf:.4f}')
st.write(f'Gradient Boosting - Mean Squared Error: {mse_gb:.4f}, R-squared: {r2_gb:.4f}')

# Predict ratings for all reviews in the dataset
df['predicted_rating_lr'] = model_lr.predict(vectorizer.transform(df['cleaned_review_text']))
df['predicted_rating_rf'] = model_rf.predict(vectorizer.transform(df['cleaned_review_text']))
df['predicted_rating_gb'] = model_gb.predict(vectorizer.transform(df['cleaned_review_text']))

# Display the actual and predicted ratings
st.write('### Actual vs Predicted Ratings (Linear Regression)')
st.write(df[['firm', 'overall_rating', 'predicted_rating_lr']].head())

st.write('### Actual vs Predicted Ratings (Random Forest)')
st.write(df[['firm', 'overall_rating', 'predicted_rating_rf']].head())

st.write('### Actual vs Predicted Ratings (Gradient Boosting)')
st.write(df[['firm', 'overall_rating', 'predicted_rating_gb']].head())

# Allow user input for prediction
user_review = st.text_area('Enter a review (pros + cons):')

if user_review:
    user_review_cleaned = preprocess_text(user_review)
    user_review_tfidf = vectorizer.transform([user_review_cleaned])
    
    prediction_lr = model_lr.predict(user_review_tfidf)  # Linear Regression
    prediction_rf = model_rf.predict(user_review_tfidf)  # Random Forest
    prediction_gb = model_gb.predict(user_review_tfidf)  # Gradient Boosting
    
    st.write(f'**Predicted Overall Rating (Linear Regression)**: {prediction_lr[0]:.2f}')
    st.write(f'**Predicted Overall Rating (Random Forest)**: {prediction_rf[0]:.2f}')
    st.write(f'**Predicted Overall Rating (Gradient Boosting)**: {prediction_gb[0]:.2f}')

# File uploader for custom CSV
uploaded_file = st.file_uploader("Upload your own CSV", type=["csv"])

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Overview")
    st.write(user_df.head())
