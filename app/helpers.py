import base64
from PIL import Image
from io import BytesIO
import cv2
import numpy as np


def convert_image_to_base64(image: Image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_str = base64.b64encode(buffer.getvalue()).decode()
    return image_str


def convert_image_to_array(image: Image):
    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    return image_array


def convert_array_to_image(image_array: np.array):
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_array)
    return image


def weaviate_img_search(client, img_str):
    """
    This function uses the nearImage operator in Weaviate.
    """
    sourceImage = {"image": img_str}

    weaviate_results = (
        client.query.get("ZalandoProduct", ["image", "product", "url"])
        .with_near_image(sourceImage, encode=False)
        .with_limit(2)
        .do()
    )

    return weaviate_results["data"]["Get"]["ZalandoProduct"]
