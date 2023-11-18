import weaviate
import os

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

# Create a new class
# TODO: Split up in two classes (one for products and one for images)
# client.schema.delete_class("ZalandoProduct")
class_obj = {
    "class": "ZalandoProduct",
    "vectorizer": "img2vec-neural",  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
    "moduleConfig": {"img2vec-neural": {"imageFields": ["image"]}},
    "properties": [
        {"name": "product", "dataType": ["text"]},
        {"name": "url", "dataType": ["text"]},
        {"name": "category", "dataType": ["text"]},
        {"name": "brand", "dataType": ["text"]},
        {"name": "color", "dataType": ["text"]},
        {"name": "price", "dataType": ["number"]},
        {"name": "image", "dataType": ["blob"]},
        {"name": "filepath", "dataType": ["text"]}
    ],
}
client.schema.create_class(class_obj)

print("The schema has been defined.")