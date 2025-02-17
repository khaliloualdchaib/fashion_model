import pandas as pd
import os
import requests
from urllib.parse import urlparse
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image
from tqdm import tqdm

IMAGE_FOLDER = "images"
TARGET_SIZE = (300, 300)

def resize_with_padding(image, target_size=TARGET_SIZE, fill_color=(255, 255, 255)):
    image.thumbnail((target_size[0], target_size[1]), Image.LANCZOS)
    new_img = Image.new("RGB", target_size, fill_color)
    paste_x = (target_size[0] - image.size[0]) // 2
    paste_y = (target_size[1] - image.size[1]) // 2
    new_img.paste(image, (paste_x, paste_y))
    return new_img

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
            img = Image.open(image_path)
            processed_img = resize_with_padding(img, TARGET_SIZE)
            processed_img.save(image_path, format="JPEG")
            return image_path
        else:
            return "Download failed"
    except Exception as e:
        return f"Error: {e}"

#This function drops some columns and rows, The download image function is also called here
def clean_data(csv_file, output_csv):

    #read csv
    df = pd.read_csv(csv_file )

    #drop all unnecessary columns
    columns_to_keep = ['barcode', 'style', 'tops_fit', 'bottoms_fit', 'sleeve_type', 'pattern', 'more_attributes', 'image_url_1', 'brand']
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

#In this function all the encoding happens here
def encode_csv(input_csv, output_csv, multi_label_columns):
    df = pd.read_csv(input_csv)
    df = pd.get_dummies(df, columns=['brand'], prefix='brand')

    for column in tqdm(multi_label_columns, desc="Encoding Multi-label Columns"):
        df[column] = df[column].apply(lambda x: x.split(';') if pd.notna(x) else [])
        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform(df[column])
        encoded_df = pd.DataFrame(encoded, columns=[f"{column}_{cls}" for cls in mlb.classes_])
        df = pd.concat([df, encoded_df], axis=1)
        df = df.drop(columns=[column])

    df.to_csv(output_csv, index=False)
