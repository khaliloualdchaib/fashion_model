import pandas as pd
import os
import requests
from urllib.parse import urlparse

IMAGE_FOLDER = "images"

def download_image(image_url):
    try:
        parsed_url = urlparse(image_url)
        image_name = os.path.basename(parsed_url.path)
        image_path = os.path.join(IMAGE_FOLDER, image_name)
        # Download and save image
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(image_path, "wb") as img_file:
                for chunk in response.iter_content(1024):
                    img_file.write(chunk)
            return image_path
        else:
            return "Download failed"
    except Exception as e:
        return f"Error: {e}"

def clean_data(csv_file, output_csv):
    #read csv
    df = pd.read_csv(csv_file )

    #drop all unnecessary columns
    columns_to_keep = ['category', 'style', 'tops_fit', 'bottoms_fit', 'sleeve_type', 'pattern', 'more_attributes', 'image_url_1', 'brand']
    df = df[columns_to_keep]

    #drop duplicate rows
    df = df.drop_duplicates()

    # There are 6 rows without style, 5 is because of the no detection category and 1 is an outlier
    df = df.dropna(subset=['style'])

    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    df["image_path"] = df["image_url_1"].apply(download_image)

    #drop image_url_1 column
    df = df.drop(columns=["image_url_1"])

    df.to_csv(output_csv, index=False)

