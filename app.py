import streamlit as st
import pandas as pd
import string  # Import the string module

# Load the data
df = pd.read_csv(r"C:\Users\ViezPC1\Pythonprojects\sentiment_aynalysis\glassdoor_reviews.csv")

# Preprocess the data (example)
df['cleaned_pros'] = df['pros'].str.lower().str.translate(str.maketrans('', '', string.punctuation))
df['cleaned_cons'] = df['cons'].str.lower().str.translate(str.maketrans('', '', string.punctuation))

# Group the data by firm
grouped_df = df.groupby('firm').agg({
    'overall_rating': 'mean',
    'pros': 'count',
    'cons': 'count'
}).reset_index()

grouped_df.rename(columns={'pros': 'num_reviews'}, inplace=True)

# Streamlit App Interface
st.title("Job Market Sentiment Analysis")

# Add filtering options
selected_firm = st.selectbox("Select a firm:", options=grouped_df['firm'].unique())

# Filter the data based on the selected firm
filtered_df = df[df['firm'] == selected_firm]

st.write(f"Displaying reviews for **{selected_firm}**")

# Implement pagination
page_size = 10
page_number = st.slider("Page Number", 1, len(filtered_df) // page_size + 1)

start_idx = (page_number - 1) * page_size
end_idx = start_idx + page_size

paginated_df = filtered_df.iloc[start_idx:end_idx]

# Display the data
st.dataframe(paginated_df)

# Optional: Add visualizations
st.bar_chart(grouped_df.set_index('firm')['overall_rating'])
