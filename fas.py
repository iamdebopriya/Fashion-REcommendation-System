import streamlit as st
import pickle
import requests
from io import BytesIO

# Function to download the pickle file from Dropbox
def download_pickle_from_dropbox(url, dest_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except requests.RequestException as e:
        st.error(f"Error downloading file: {e}")
        return False

# Function to load the pickle file
@st.cache_data
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
        st.error(f"Error unpickling the file: {e}")
        return None

# Dropbox direct download URL (replace with your actual URL)
dropbox_url = 'https://www.dropbox.com/scl/fi/weuku2nwesbm5a1o1y49z/recommendation_model.pkl?rlkey=ejuu4v21jpjl4mtckzyg8bufx&st=hx7zllpv&dl=1'

# Path to save the pickle file temporarily
pickle_file_path = 'recommendation_model.pkl'

# Download the pickle file from Dropbox
download_success = download_pickle_from_dropbox(dropbox_url, pickle_file_path)

if download_success:
    # Load the pickle file
    model = load_pickle(pickle_file_path)

    if model is None:
        st.error("Failed to load model. Please check the file path and format.")
    else:
        # Extract the DataFrame and cosine similarity matrix
        df = model.get('df')
        cosine_sim = model.get('cosine_sim')

        if df is None or cosine_sim is None:
            st.error("Model does not contain the expected data.")
        else:
            # TF-IDF Vectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df['combined_features'])

            # Function to get recommendations based on description
            @st.cache_data
            def get_recommendations_from_description(description):
                description_tfidf = tfidf.transform([description])
                sim_scores = cosine_similarity(description_tfidf, tfidf_matrix)
                sim_scores = sim_scores.flatten()
                sim_scores = list(enumerate(sim_scores))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:6]
                item_indices = [i[0] for i in sim_scores]
                return df.iloc[item_indices]

            # Streamlit App
            st.title("Fashion Product Recommendation System")

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

            description = st.text_area("Enter product description")

            if st.button("Get Recommendations"):
                if description:
                    recommendations = get_recommendations_from_description(description)
                    st.write("Recommended products:")
                    for _, row in recommendations.iterrows():
                        image_url = row['img']
                        try:
                            image_response = requests.get(image_url, timeout=5)
                            image_response.raise_for_status()
                            st.image(BytesIO(image_response.content), caption=row['name'])
                        except requests.HTTPError:
                            st.write(f"Image not found for {row['name']}")
                        except requests.RequestException as e:
                            st.write(f"Error fetching image: {e}")

                        st.write(f"Description: {row['description']}")
                else:
                    st.write("Please enter a description.")
else:
    st.error("Failed to download or load the pickle file.")
