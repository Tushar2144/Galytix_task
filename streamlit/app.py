import gensim
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the Word2Vec model from directory or model path
model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/paltu/anaconda_copy/GoogleNews-vectors-negative300.bin', binary=True)

# Load phrases from the CSV file (assumes a 'phrase' column in the CSV) from directory or csv_file_path
phrases_df = pd.read_csv("C:/Users/paltu/anaconda_copy/phrases.csv", encoding='latin1')
phrases = phrases_df['Phrases'].tolist()

# Assign Word2Vec embeddings to each word in the phrases
def get_phrase_embedding(phrase, model):
    words = phrase.split()
    valid_words = [word for word in words if word in model]
    if valid_words:
        phrase_embedding = np.mean([model[word] for word in valid_words], axis=0)
        return phrase_embedding
    else:
        return None

# Create a dictionary to store phrase embeddings
phrase_embeddings = {}
for phrase in phrases:
    embedding = get_phrase_embedding(phrase, model)
    if embedding is not None:
        phrase_embeddings[phrase] = embedding

# Batch execution: Calculate cosine similarities between all pairs of phrases
def calculate_similarity_matrix(phrase_embeddings, phrases):
    similarity_matrix = np.zeros((len(phrases), len(phrases)))
    for i in range(len(phrases)):
        for j in range(len(phrases)):
            similarity = cosine_similarity([phrase_embeddings[phrases[i]]], [phrase_embeddings[phrases[j]]])[0][0]
            similarity_matrix[i][j] = similarity
    return similarity_matrix

similarity_matrix = calculate_similarity_matrix(phrase_embeddings, phrases)

# On-the-fly execution: Find the closest match to a user-input phrase
def find_closest_match(user_input, phrase_embeddings, phrases):
    user_embedding = get_phrase_embedding(user_input, model)
    if user_embedding is not None:
        similarities = [cosine_similarity([user_embedding], [phrase_embeddings[phrase]])[0][0] for phrase in phrases]
        closest_match_idx = np.argmax(similarities)
        closest_match = phrases[closest_match_idx]
        distance = 1 - similarities[closest_match_idx]
        return closest_match, distance
    else:
        return "User input has no valid words for embeddings.", None

# Streamlit UI
st.title("Word2Vec Phrase Similarity App")

# Input text box for user's input
user_input = st.text_area("Enter a phrase:", "I need help with data analysis")

# Button to find closest match
# if st.button("Find Closest Match"):
#     closest_match, distance = find_closest_match(user_input, phrase_embeddings, phrases)
#     # st.write(f"Closest Match: {closest_match}, Distance: {distance:.4f}")

# or use below code

# Button to find the closest match
if st.button("Find Closest Match"):
    closest_match, distance = find_closest_match(user_input, phrase_embeddings, phrases)
    st.write("##  **Closest Match:**", closest_match)
    st.write("##  **Euclidean Distance:**", f"{distance:.4f}")


