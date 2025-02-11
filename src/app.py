from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import joblib

# Load the saved model
ingred_matrix = joblib.load("../model/ingred_matrix.pkl")
data = pd.read_csv("../data/products.csv")

# Initialize FastAPI app
app = FastAPI(title="Product Recommendation API")

# Recommender function
def recommender(search: str, num_recommendations: int = 5):
    cs_list = []
    brands = {}
    output = []

    # Check if the product exists
    if search not in data['product_name'].values:
        return {"error": "Product not found"}

    # Get index of searched product
    idx = data[data['product_name'] == search].index.item()
    point1 = np.array(ingred_matrix.iloc[idx][1:], dtype=np.float32)
    prod_type = data.loc[idx, 'product_type']
    brand_search = data.loc[idx, 'brand']

    # Filter by product type
    data_by_type = data[data['product_type'] == prod_type].copy()

    # Compute cosine similarity
    for j, row in data_by_type.iterrows():
        point2 = np.array(ingred_matrix.iloc[j][1:], dtype=np.float32)
        cos_sim = np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))
        cs_list.append(cos_sim)

    # Add similarity scores
    data_by_type["cos_sim"] = cs_list
    data_by_type = data_by_type.sort_values("cos_sim", ascending=False)

    # Exclude the input product itself
    data_by_type = data_by_type[data_by_type["product_name"] != search]

    # Select top recommendations ensuring brand diversity
    for _, row in data_by_type.iterrows():
        brand = row["brand"]
        if brand != brand_search and brands.get(brand, 0) < 2:
            brands[brand] = brands.get(brand, 0) + 1
            output.append({
                "product_name": row["product_name"],
                "cosine_similarity": row["cos_sim"],
                "price": row["price"],
                "url": row["product_url"]
            })
        if len(output) >= num_recommendations:
            break

    return output

# API Endpoint
@app.get("/recommend", summary="Get product recommendations")
async def recommend(product_name: str = Query(..., description="Enter product name")):
    recommendations = recommender(product_name, num_recommendations=5)
    return recommendations

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
