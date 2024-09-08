# Glassdoor Reviews Sentiment Analysis

This repository contains a Streamlit application for analyzing Glassdoor reviews. The app performs sentiment analysis on company reviews, provides data visualizations, and allows for predictive modeling of company ratings based on review text.

## Features

- **Data Overview:** Displays the first few rows of the dataset and provides an overview of the data.
- **Search Functionality:** Allows users to search for companies and view relevant data.
- **Filtering:** Apply filters based on overall rating, number of reviews, and counts of pros and cons.
- **Top Firms:** Displays top firms by average rating and number of reviews.
- **Word Clouds:** Generates word clouds for pros and cons reviews.
- **Predictive Modeling:** Predicts company ratings based on user-provided review text using multiple models.
- **Custom CSV Upload:** Upload your own CSV to view and analyze data.

## Requirements

Ensure you have the following installed:

- Python 3.x
- Streamlit
- Pandas
- NLTK
- Scikit-learn
- WordCloud
- Joblib
- Matplotlib
- Seaborn

You can install the required Python packages using pip:

```bash
pip install -r requirements.txt
