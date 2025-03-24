import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import resnet101, ResNet101_Weights
import torch.nn as nn
import time  # For loading animation

st.set_page_config(page_title="Skincare AI", layout="wide")
# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./data/face_products2.csv")
    
    # Remove '£' and convert price column to numeric
    df["price"] = df["price"].astype(str).str.replace("£", "", regex=False).astype(float)
    
    return df

df = load_data()

# Page Config

# Custom CSS for better UI
st.markdown("""
    <style>
        body { background-color: #121212; color: white; }
        .uploadedFile { background-color: #1E1E1E; padding: 10px; border-radius: 10px; }
        .sidebar .stButton { background-color: #31C48D; color: white; }
        h3 { color: #31C48D; }
    </style>
""", unsafe_allow_html=True)

# Sidebar - User Inputs
st.sidebar.title("🧴 Skincare Analysis")
uploaded_file = st.sidebar.file_uploader("Upload Your Skin Image", type=["jpg", "jpeg", "png"])
manual_skin_type = st.sidebar.selectbox("Select Skin Type (Optional)", ["Auto Detect", "Oily", "Dry", "Normal"])
price_filter = st.sidebar.slider("Filter by Price Range ($)", float(df["price"].min()), float(df["price"].max()), (10.0, 80.0))

# Load ML models
skin_type_model = torch.load('models/skin_type_model_complete.pth', map_location=torch.device('cpu'))
skin_type_model.eval()
concern_model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
num_ftrs = concern_model.fc.in_features
concern_model.fc = nn.Linear(num_ftrs, 4)
concern_model.load_state_dict(torch.load("./notebooks/best_model_concern___.pth"))
concern_model.eval()

# Load product data
skincare_products_df = pd.read_csv('./data/face_products2.csv').dropna(subset=['Concern List_'])
skincare_products_df[['Skin Type']] = skincare_products_df[['Skin Type']].fillna('Normal')

# Remove '£' from price and convert to float
skincare_products_df["price"] = skincare_products_df["price"].astype(str).str.replace("£", "", regex=False).astype(float)

# One-hot encoding for skin type
encoder_skin = OneHotEncoder(sparse_output=False)
skin_type_encoded = encoder_skin.fit_transform(skincare_products_df[['Skin Type']])
skin_type_df = pd.DataFrame(skin_type_encoded, columns=encoder_skin.categories_[0])

# TF-IDF for ingredient similarity
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
ingredients_tfidf = tfidf_vectorizer.fit_transform(skincare_products_df['ingredients'].fillna(''))

# Image Preprocessing
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# Function for product recommendation
def recommend_products(user_skin_type, user_concern, ingredients_tfidf):
    user_skin_type_encoded = encoder_skin.transform([[user_skin_type]])

    # Create concern vector
    all_concerns = ['Acne', 'Bags', 'Enlarged pores', 'Redness']
    user_concern_vector = np.zeros(len(all_concerns))
    for concern in user_concern.split(', '):
        if concern in all_concerns:
            user_concern_vector[all_concerns.index(concern)] = 1

    # Compute similarity scores
    final_similarity = cosine_similarity(user_skin_type_encoded, skin_type_df) * 0.5 + \
                       cosine_similarity(tfidf_vectorizer.transform([user_concern]), ingredients_tfidf) * 0.5

    # Get top recommendations
    recommended_indices = final_similarity.argsort()[0][-5:][::-1]
    recommended_products = skincare_products_df.iloc[recommended_indices]

    return recommended_products[
        (recommended_products["price"] >= price_filter[0]) & 
        (recommended_products["price"] <= price_filter[1])
    ][['product_name', 'product_url', 'product_type', 'price', 'image_url']]

# Main UI Section
st.title("Skincare Recommendations")
st.write("Upload an image to analyze your skin type and get personalized product recommendations.")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image.resize((300, 400)), caption="Uploaded Image", use_container_width=False)

    # Processing Animation
    with st.spinner('Analyzing your skin...'):
        time.sleep(3)  # Simulating processing time
        image_input = preprocess_image(image)

        # Predict skin type
        skin_type_labels = ["Oily", "Dry", "Normal"]
        skin_type_prediction = torch.argmax(skin_type_model(image_input), dim=1).item()
        skin_type = manual_skin_type if manual_skin_type != "Auto Detect" else skin_type_labels[skin_type_prediction]
        st.success(f"✅ Predicted Skin Type: {skin_type}")

        # Predict concern
        concern_prediction = torch.argmax(concern_model(image_input), dim=1).item()
        concern_labels = ['Acne', 'Bags', 'Enlarged pores', 'Redness']
        concern = concern_labels[concern_prediction]
        st.warning(f"⚠️ Detected Concern: {concern}")

    # Recommendations
    st.write("### 🔥 Recommended Products for You")
    recommended_products = recommend_products(skin_type, concern, ingredients_tfidf)
    cols = st.columns(3)  # Display in 3 columns for better UI

    for idx, (_, row) in enumerate(recommended_products.iterrows()):
        col = cols[idx % 3]
        with col:
            st.markdown(f"""
                <div style="background-color: #1E1E1E; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 15px;">
                    <img src="{row['image_url'] if pd.notna(row['image_url']) else 'https://via.placeholder.com/200'}" 
                         width="150" style="border-radius: 10px;" />
                    <h3>{row['product_name']}</h3>
                    <p><strong>Type:</strong> {row['product_type']}</p>
                    <p style="color: #31C48D; font-size: 20px; font-weight: bold;">💲{row['price']}</p>
                    <a href="{row['product_url']}" target="_blank">
                        <button style="background-color: #31C48D; padding: 10px 15px; border-radius: 5px; cursor: pointer;">
                            View Product
                        </button>
                    </a>
                </div>
            """, unsafe_allow_html=True)
