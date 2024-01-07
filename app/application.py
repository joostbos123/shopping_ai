from flask import Flask, render_template, request
from PIL import Image
import weaviate
import os

from helpers import (
    convert_image_to_base64,
    convert_image_to_array,
    convert_array_to_image,
    weaviate_img_search,
)
from Segmentation import Segmentation

CLASS_MAPPING = {
    "t-shirt": "t-shirt",
    # "trouser": "trouser",
    # "pullover": "pullover",
    # "dress": "dress",
    # "coat": "coat",
    # "shirt": "shirt",
    # "shoes": "shoes",
    # "bag": "bag",
    "glasses": "glasses",
}

SEGMENTATION = False
if SEGMENTATION:
    model = Segmentation(CLASS_MAPPING)

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
if not WEAVIATE_URL:
    WEAVIATE_URL = "http://localhost:8080"

# creating the application and connecting it to the Weaviate local host
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "/temp_images"
client = weaviate.Client(WEAVIATE_URL)


if client.is_ready():
    # Defining the pages that will be on the website
    @app.route("/")
    def home():  # home page
        return render_template("index.html")

    @app.route(
        "/process_image", methods=["POST"]
    )  # save the uploaded image and convert it to base64

    # process the image upload request by converting it to base64 and querying Weaviate
    def process_image():
        image_uploaded = Image.open(request.files["filepath"].stream)

        image_array = convert_image_to_array(image_uploaded)
        image_str = convert_image_to_base64(image_uploaded)

        if SEGMENTATION:
            detections = model.predict(
                image_array, box_threshold=0.45, text_threshold=0.35
            )
            items = model.get_images_per_class(image_array, detections)
        else:
            items = [{"class": "", "image": image_array, "polygon": []}]

        content = list()
        for item in items:
            class_name = item["class"]
            search_image_array = item["image"]
            polygon = item["polygon"]

            # Convert image to base64
            search_image = convert_array_to_image(search_image_array)
            search_image_str = convert_image_to_base64(search_image)

            # Search for similar products
            search_results = weaviate_img_search(client, search_image_str)

            content.append(
                {
                    "class": class_name,
                    "polygon": polygon,
                    "results":
                    [
                        {
                        "image_str": result["image"],
                        "product": result["product"],
                        "url": result["url"],
                        }
                        for result in search_results
                    ]
                }
            )

        print(f"\n {content} \n")
        return render_template(
            "index.html",
            content=content,
            search_image=image_str,
        )

else:
    print("There is no Weaviate Cluster Connected.")

# run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5050", debug=True)
