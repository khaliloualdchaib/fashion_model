import pandas as pd

# Load the dataset
df = pd.read_csv("data/sampled_data.csv")


# Body type rules
body_type_rules = {
    "hourglass": {
        "category": ["dress", "blazer", "tops", "pencil skirt", "high-waist trousers"],
        "style": ["sophisticated", "feminine", "minimalism"],
        "colors": ["black", "navy", "red", "emerald-green", "bordeaux"],
        "pattern": ["plain", "floral", "vertical-stripes"],
        "occasion": ["semi-formal", "office", "dating"],
        "tops_fit": ["bodycon-tops", "fitted-tops"],
        "waist_type": ["high-waist"],
        "bottoms_fit": ["slim-bottoms", "skinny-bottoms"],
        "neckline_type": ["v-neck", "square-neck", "sweetheart-neck"],
        "sleeve_type": ["straight-sleeve", "puff-sleeve"],
    },
    "apple": {
        "category": ["wrap dress", "flowy tops", "cardigan", "jumpsuit"],
        "style": ["minimalism", "sophisticated", "casual"],
        "colors": ["dark shades", "black", "navy", "burgundy"],
        "pattern": ["vertical-stripes", "plain"],
        "occasion": ["casual", "semi-formal", "winter"],
        "tops_fit": ["regular-fit-tops", "oversized-tops"],
        "waist_type": ["mid-waist"],
        "bottoms_fit": ["straight-fit-bottoms"],
        "neckline_type": ["v-neck", "round-neck"],
        "sleeve_type": ["dolman-sleeve", "straight-sleeve"],
    },
    "rectangle": {
        "category": ["peplum tops", "blazer", "ruffled dress", "belted dress"],
        "style": ["feminine", "business", "sophisticated"],
        "colors": ["bright colors", "pink", "red", "mustard"],
        "pattern": ["geometric", "floral", "abstract"],
        "occasion": ["office", "semi-formal", "dating"],
        "tops_fit": ["fitted-tops", "peplum-tops"],
        "waist_type": ["belted"],
        "bottoms_fit": ["flare-bottoms", "wide-bottoms"],
        "neckline_type": ["v-neck", "square-neck"],
        "sleeve_type": ["puff-sleeve", "flare-sleeve"],
    }
}

def score_item(item, selected_body_type):
    score = 0
    selected_rules = body_type_rules[selected_body_type]

    conflicting_rules = {key: set() for key in selected_rules.keys()}
    for body_type, rules in body_type_rules.items():
        if body_type != selected_body_type:
            for key, values in rules.items():
                conflicting_rules[key].update(values)
    
    for key, values in selected_rules.items():
        if isinstance(item[key], str):
            item_values = set(item[key].replace(";", ",").split(", ")) 
            positive_matches = item_values & set(values)
            negative_matches = item_values & conflicting_rules[key]
            score += len(positive_matches)
            score -= len(negative_matches)
    
    return score

def recommend_clothing(body_type, top_n=10):
    if body_type not in body_type_rules:
        return {"error": "Invalid body type. Choose from 'hourglass', 'apple', or 'rectangle'."}

    df["score"] = df.apply(lambda item: score_item(item, body_type), axis=1)
    recommended_items = df.sort_values(by="score", ascending=False).head(top_n)

    return {"recommended_items": recommended_items["barcode"].tolist()}
