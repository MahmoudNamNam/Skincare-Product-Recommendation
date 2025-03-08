from fuzzywuzzy import process

def find_closest_product_link(search, data):
    choices = data["product_url"].tolist()
    best_match, score = process.extractOne(search, choices)
    if score > 80: 
        return best_match
    return None