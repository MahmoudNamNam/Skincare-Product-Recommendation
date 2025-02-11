# Skincare Product Recommendation

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the API

### 1. Start the API

```bash
python app.py
```

The API will be available at: `http://127.0.0.1:8000`

### 2. Make a Request

Use the following format to get recommendations:

```bash
http://127.0.0.1:8000/recommend?product_name=The%20Ordinary%20Squalane%20Cleanser%2050ml
```

## API Endpoints

### `GET /recommend`

- **Description:** Returns recommended products similar to the given product name.
- **Query Parameter:** `product_name` (string) - The name of the product to search for.
- **Response:** JSON object containing recommended products.

**Example Response:**

```json
[
    {
        "product_name": "La Roche-Posay Toleriane Dermo-Cleanser 200ml",
        "cosine_similarity": 0.23094011843204498,
        "price": "£12.50",
        "url": "https://www.lookfantastic.com/la-roche-posay-toleriane-dermo-cleanser-200ml/11091885.html"
    },
    {
        "product_name": "ESPA Optimal Skin ProCleanser 100ml",
        "cosine_similarity": 0.22135944664478302,
        "price": "£32.00",
        "url": "https://www.lookfantastic.com/espa-optimal-skin-pro-cleanser/12226523.html"
    },
]
```
