import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import requests

# Load the pickle file directly
@st.cache_resource
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Path to the pickle file (update this path as needed)
pickle_file_path = 'recommendation_model.pkl'

# Load the pickle file
model = load_pickle(pickle_file_path)

# Extract the DataFrame and cosine similarity matrix
df = model['df']
cosine_sim = model['cosine_sim']

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Function to get recommendations based on description
@st.cache_data
def get_recommendations_from_description(description):
    description_tfidf = tfidf.transform([description])
    sim_scores = cosine_similarity(description_tfidf, tfidf_matrix)  # Compute similarity with tfidf_matrix
    sim_scores = sim_scores.flatten()
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar items
    item_indices = [i[0] for i in sim_scores]
    return df.iloc[item_indices]

# Streamlit App
st.title("Fashion Product Recommendation System")

# Apply custom CSS for light gradient background
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #f9c2c2, #c2d9ff); /* Light pink to light blue gradient */
    }
    .stTitle {
        color: #007bff; /* Title color */
        font-size: 2em; /* Title size */
    }
    .stButton > button {
        background-color: #007bff; /* Button background color */
        color: #fff; /* Button text color */
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #0056b3; /* Button hover color */
    }
    </style>
""", unsafe_allow_html=True)

# Input for product description
description = st.text_area("Enter product description")

if st.button("Get Recommendations"):
    if description:
        recommendations = get_recommendations_from_description(description)
        st.write("Recommended products:")
        for _, row in recommendations.iterrows():
            # Show image
            image_url = row['img']
            try:
                image_response = requests.get(image_url, timeout=5)
                image_response.raise_for_status()
                st.image(BytesIO(image_response.content), caption=row['name'])
            except requests.HTTPError:
                st.write(f"Image not found for {row['name']}")
            except requests.RequestException as e:
                st.write(f"Error fetching image: {e}")

            # Show product description
            st.write(f"Description: {row['description']}")
    else:
        st.write("Please enter a description.")
