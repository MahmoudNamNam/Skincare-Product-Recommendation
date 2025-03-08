from fuzzywuzzy import process

def find_closest_product_name(search, data):
    choices = data["product_name"].tolist()
    best_match, score = process.extractOne(search, choices)
    if score > 80: 
        return best_match
    return None