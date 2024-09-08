python -m venv aimlpratice                                                                                                  
 .\aimlpratice\Scripts\Activate   
 pip install pandas matplotlib seaborn nltk vaderSentiment textblob transformers                                             
pip install transformers torch   
pip install streamlit pandas numpy
streamlit run app.py                  

Glassdoor Reviews Sentiment Analysis
This repository contains a Streamlit application for analyzing Glassdoor reviews. The app performs sentiment analysis on company reviews, provides data visualizations, and allows for predictive modeling of company ratings based on review text.

Features
Data Overview: Displays the first few rows of the dataset and provides an overview of the data.
Search Functionality: Allows users to search for companies and view relevant data.
Filtering: Apply filters based on overall rating, number of reviews, and counts of pros and cons.
Top Firms: Displays top firms by average rating and number of reviews.
Word Clouds: Generates word clouds for pros and cons reviews.
Predictive Modeling: Predicts company ratings based on user-provided review text using multiple models.
Custom CSV Upload: Upload your own CSV to view and analyze data.
Requirements
Ensure you have the following installed:

Python 3.x
Streamlit
Pandas
NLTK
Scikit-learn
WordCloud
Joblib
Matplotlib
Seaborn
You can install the required Python packages using pip:

bash
Copy code
pip install streamlit pandas nltk scikit-learn wordcloud joblib matplotlib seaborn
Setup
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/repository-name.git
cd repository-name
Download NLTK Data: Make sure to download the necessary NLTK stopwords data:

python
Copy code
import nltk
nltk.download('stopwords')
Prepare Your Data: Place your CSV file containing Glassdoor reviews in the project directory. Ensure the CSV file has columns pros, cons, overall_rating, firm, and any other relevant columns.

Run the Streamlit App: Use the following command to run the app:

bash
Copy code
streamlit run app.py
Access the App: Open a web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).

Usage
Data Overview: View the initial rows of the dataset and grouped data by firm.
Search: Use the search bar to find specific companies.
Filtering: Use the sidebar sliders to filter data based on rating, number of reviews, pros, and cons.
Top Firms: View the top firms by rating and number of reviews, and visualize them using bar charts.
Word Clouds: See the most frequent words in pros and cons reviews through word clouds.
Predictive Modeling: Enter a review to get predictions of the overall rating from different models.
Upload Custom CSV: Upload your own CSV file to analyze custom data.
Model Training
The app uses pre-trained models for prediction. If the models are not found, they will be trained on the dataset. The trained models are saved using joblib and will be loaded in subsequent runs.

Notes
Ensure that the CSV file structure matches the expected format.
If you encounter performance issues, consider optimizing data processing or model loading as described in the code comments.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to modify any part of the README to better fit your specific project details or preferences!
