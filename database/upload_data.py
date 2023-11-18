import weaviate
import pandas as pd
import base64
import os

PATH_DATA_IMAGES_FOLDER = "../zalando_scraper/data/images"
PATH_DATA_PRODUCTS = "../zalando_scraper/data/products.csv"

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
if not WEAVIATE_URL:
    WEAVIATE_URL = "http://localhost:8080"

client = weaviate.Client(
    url=WEAVIATE_URL,
    # additional_headers={
    #     "X-OpenAI-Api-Key": "API KEY",
    # }  # Replace with your API key
    # auth_client_secret=auth_config,
)

df_products = pd.read_csv(PATH_DATA_PRODUCTS, nrows=1000)

client.batch.configure(
    batch_size=100,
    dynamic=True,
    timeout_retries=3,
    callback=None,
)  # Configure batch

with client.batch as batch:  # Initialize a batch process
    for index_product, row in df_products.iterrows():  # Batch import data
        
        print(f"importing product: {index_product + 1}")
        for image in os.listdir(f"{PATH_DATA_IMAGES_FOLDER}/product_{'%04d'%row['id']}"):

            # Read image and convert to base64
            image_path = f"{PATH_DATA_IMAGES_FOLDER}/product_{'%04d'%row['id']}/{image}"
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

            properties = {
                "product": row["name"],
                "url": row["url"],
                "category": row["category"],
                "brand": row["brand"],
                "color": row["color"],
                "price": float(row["price"].replace(",", ".")),
                "image": encoded_string,
                "filepath": image_path,
            }
            batch.add_data_object(data_object=properties, class_name="ZalandoProduct")
