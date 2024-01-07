import weaviate
import base64
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

# Search for similar products
search_image_path = "../zalando_scraper/data/images/product_0000/image_01.jpg"
with open(search_image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
search_image = {"image": encoded_string}
weaviate_results = (
    client.query.get("ZalandoProduct", ["image", "product", "url", "filepath"])
    .with_near_image(search_image, encode=False)
    .with_limit(2)
    .do()
)