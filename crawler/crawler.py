import requests
from bs4 import BeautifulSoup
import queue
import re
import time
import random
import threading
from threading import Thread


class Crawler:
    def __init__(self) -> None:
        self.config = {
            "name": "Bruna",
            "url": "https://www.bruna.nl/boeken",
            "product_fields": {
                "name": {"selector": ".header-pdp", "required": True},
                "description": {"selector": ".description", "required": True},
                "category": {"selector": ".breadcrumb__link", "element": 2},
                "brand": {},
                "color": {},
                "price": {"selector": ".huge.colored.heavy"},
                "image_urls": {"selector": ".product-carrousel"},
            },
        }

        self.urls = queue.Queue()
        self.urls.put(self.config["url"])
        self.visited_urls = []
        self.products = []
        self.num_workers = 5

    def backoff_delay(self, backoff_factor, attempts):
        # backoff algorithm
        delay = backoff_factor * (2**attempts)
        return delay

    def retry_request(
        self, url, max_retries=4, status_forcelist=[429, 500, 502, 503, 504], **kwargs
    ):
        # Make number of requests required
        for retry_num in range(max_retries):
            try:
                response = requests.get(url, **kwargs)
                if response.status_code in status_forcelist:
                    # Retry request
                    self.backoff_delay(0.5, retry_num)
                    continue
                return response
            except requests.exceptions.ConnectionError:
                pass
        return None

    def get_html_content(self, url):
        response = self.retry_request(url)
        return response.content

    def crawl_page(self, soup, url):
        link_elements = soup.select("a[href]")
        for link_element in link_elements:
            url = link_element["href"]

            if url.startswith(self.config["url"]) and url not in self.visited_urls:
                self.urls.put(url)
                self.visited_urls.append(url)

    def scrape_page(self, soup, url):
        product = {}
        product["url"] = url

        try:
            for field, field_config in self.config["product_fields"].items():
                if field_config:
                    # Scrape image urls
                    if field == "image_urls":
                        product["image_urls"] = []
                        # Find element with the specified selector
                        image_el = soup.select_one(field_config["selector"])
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
                            product[field] = soup.select_one(
                                field_config["selector"]
                            ).text

                    # Only push product if all information is available
                    if not product[field] and field_config.get("required"):
                        print(f"Missing field {field} when craping page {url}")
                        return
                    self.products.append(product)

        except Exception as e:
            if field_config.get("required"):
                print(
                    f"Not able to scrape {field} information from the page {url}, caused by error {e}"
                )
                return

    def queue_worker(self):
        while not self.urls.empty():
            current_url = self.urls.get()
            content = self.get_html_content(current_url)
            soup = BeautifulSoup(content, "html.parser")

            self.crawl_page(soup, current_url)

            self.scrape_page(soup, current_url)
            self.urls.task_done()

            print(threading.currentThread().ident)

            time.sleep(random.uniform(0, 1))

    def run(self):
        start_time = time.time()

        for i in range(self.num_workers):
            Thread(target=self.queue_worker, daemon=True).start()
            time.sleep(1)
        # self.queue_worker()
        self.urls.join()

        end_time = time.time() - start_time
        print(f"Crawling {len(self.products)} products from {self.config['name']} took: {end_time / 60} minutes")


crawler = Crawler()
crawler.run()
