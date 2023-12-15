import requests
from bs4 import BeautifulSoup
import queue
import re
import time
import random

config = {
    "name": "bookspot",
    "url": "https://www.bookspot.nl/boeken",
    "product_fields": {
        "name": {"selector": ".header-pdp"},
        "description": {"selector": ".description"},
        "category": {"selector": ".breadcrumb__link", "element": 2},
        "brand": {},
        "color": {},
        "price": {"selector": ".huge.colored.heavy"},
        "image_urls": {"selector": ".bookflip"},
    },
}


def crawl_page(soup, url, visited_urls, urls):
    link_elements = soup.select("a[href]")
    for link_element in link_elements:
        url = link_element["href"]

        if url.startswith(config["url"]):
            if url not in visited_urls and url not in [item[1] for item in urls.queue]:
                urls.put(url)


def get_html(url):
    try:
        return requests.get(url).content
    except Exception as e:
        print(e)
        return ""


def scrape_page(soup, url, products):
    product = {}
    product["url"] = url

    try:
        for field, field_config in config["product_fields"].items():
            # Scrape image urls
            if field == "image_urls":
                product["image_urls"] = []
                # Loop over all elements with the specified selector
                for image_el in soup.select(".bookflip"):
                    # Loop over all image elements
                    for image_img in image_el.find_all("img"):
                        # Store the src of the image in the list of image urls
                        product["image_urls"].append(image_img["src"])

            # Scrape text fields
            else:
                if field_config.get("element"):
                    product[field] = soup.select(field_config["selector"])[
                        field_config["element"]
                    ].text
                # Use first element if no element is specified
                else:
                    product[field] = soup.select_one(field_config["selector"])

            # Only push product if all information is available
            if not product[field]:
                raise Exception(f"Missing field {field} when craping page {url}")
            products.append(product)

    except Exception as e:
        print(
            f"Not able to scrape all information from the page {url}, caused by error {e}"
        )


def main():
    urls = queue.Queue()
    urls.put(config["url"])
    visited_urls = []
    products = []

    while not urls.empty():
        current_url = urls.get()
        soup = BeautifulSoup(get_html(current_url), "html.parser")

        visited_urls.append(current_url)
        crawl_page(soup, current_url, visited_urls, urls)

        scrape_page(soup, current_url, products)
    time.sleep(random.uniform(1, 3))
