import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load the model
skin_type_model = torch.load('models/skin_type_model_complete.pth', map_location=torch.device('cpu'))
skin_type_model.eval()


concern_model= torch.load('models/concern_model_complete.pth', map_location=torch.device('cpu'))
concern_model.eval() 

# Load the product data
skincare_products_df = pd.read_csv('./data/face_products.csv')
skincare_products_df = skincare_products_df.dropna(subset=['Concern List_'])

# Preprocess for recommendation system
skincare_products_df[['Skin Type']] = skincare_products_df[['Skin Type']].fillna('Normal')
encoder_skin = OneHotEncoder(sparse_output=False)

# One-hot encode the 'Skin Type' column
skin_type_encoded = encoder_skin.fit_transform(skincare_products_df[['Skin Type']])

# Get the column names based on the unique categories
skin_type_columns = encoder_skin.categories_[0]

# Create a DataFrame with the correct column names
skin_type_df = pd.DataFrame(skin_type_encoded, columns=skin_type_columns)
all_concerns = ['Acne', 'Bags', 'Enlarged pores', 'Redness']
concern_df = pd.DataFrame(0, index=skincare_products_df.index, columns=all_concerns)

# Fill concern_df based on the 'Concern List_'
for idx, concerns in enumerate(skincare_products_df['Concern List_']):
    if concerns:
        for concern in concerns.split(', '):
            if concern in concern_df.columns:
                concern_df.at[idx, concern] = 1

# TF-IDF for ingredients
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
ingredients_tfidf = tfidf_vectorizer.fit_transform(skincare_products_df['ingredients'])

# Recommendation function
def recommend_products(user_skin_type, user_concern, ingredients_tfidf):
    # Encode user input (skin type and concern) into the same format
    user_skin_type_encoded = encoder_skin.transform([[user_skin_type]])
    
    # Encode user concern into the same format
    user_concern_vector = np.zeros(len(all_concerns))
    for concern in user_concern.split(', '):
        if concern in all_concerns:
            user_concern_vector[all_concerns.index(concern)] = 1

    # Combine user input (skin type and concern) into a single feature vector
    user_vector = np.concatenate([user_skin_type_encoded.flatten(), user_concern_vector])

    # Calculate cosine similarity with each product's features (skin type and concern)
    skin_similarity = cosine_similarity([user_vector[:len(user_skin_type_encoded.flatten())]], skin_type_df)
    concern_similarity = cosine_similarity([user_vector[len(user_skin_type_encoded.flatten()):]], concern_df)

    # Calculate the final similarity score by combining skin and concern similarities (you can adjust the weight)
    combined_similarity = 0.3 * skin_similarity + 0.7 * concern_similarity

    # Calculate similarity with ingredients using TF-IDF cosine similarity
    user_ingredients_tfidf = tfidf_vectorizer.transform([user_concern])
    ingredient_similarity = cosine_similarity(user_ingredients_tfidf, ingredients_tfidf)

    # Combine all similarities
    final_similarity = 0.7 * combined_similarity + 0.3 * ingredient_similarity

    # Get the top 5 recommended products
    recommended_indices = final_similarity.argsort()[0][-5:][::-1]  # Top 5 products based on similarity

    recommended_products = skincare_products_df.iloc[recommended_indices]

    return recommended_products[['product_name', 'product_url', 'product_type', 'price']]

# Helper function for image preprocessing (for both models)
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

st.title("Skincare Product Recommendation System")

uploaded_file = st.file_uploader("Upload an image of your skin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Resize image before displaying
    image_resized = image.resize((300, 400))  # Resize to 200x200 pixels (adjust as needed)
    st.image(image_resized, caption='Uploaded Skin Image', use_container_width  =False)

    skin_type_labels = ["Oily", "Dry", "Normal"] 

    # Predict skin type
    image_input = preprocess_image(image)
    with torch.no_grad():
        skin_type_output = skin_type_model(image_input)
        skin_type_prediction = torch.argmax(skin_type_output, dim=1).item()

    # Map index to actual label
    skin_type = skin_type_labels[skin_type_prediction]

    
    st.write(f"Predicted Skin Type: {skin_type}")

    with torch.no_grad():
        concern_output = concern_model(image_input)
        concern_prediction = torch.argmax(concern_output, dim=1).item()
        concern = all_concerns[concern_prediction]
    
    st.write(f"Predicted Concern: {concern}")

    recommended_products = recommend_products(skin_type, concern, ingredients_tfidf)
    st.write("Recommended Products:")
    for index, row in recommended_products.iterrows():
        st.markdown(f"**{row['product_name']}**")
        st.markdown(f"[Product Link]({row['product_url']})")
        st.write(f"Type: {row['product_type']}, Price: {row['price']}")
