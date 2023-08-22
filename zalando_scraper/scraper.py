import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os

ZALANDO_HOME_PAGE = 'https://www.zalando.nl'

ZALANDO_PRODUCT_PAGES = {
    'dameskleding': [
        'dameskleding-jurken',
        'dameskleding-shirts',
        'dameskleding-broeken',
        'dameskleding-jeans',
        'dameskleding-bloezen',
        'dameskleding-jassen',
        'dameskleding-badmode',
        'dameskleding-pullovers-sweaters',
        'dameskleding-rokken',
        'dames-vesten-breiwerk',
        'sportkleding',
        'dameskleding-broeken-shorts',
        'dameskleding-broeken-overalls-jumpsuits',
        'lange-jassen-dames',
        'dameskleding-ondergoed',
        'dameskleding-ondergoed-nachtkleding',
        'dameskleding-kousen'
    ],
    'damesschoenen': [
        'damesschoenen-sneakers',
        'damesschoenen-sandalen',
        'damesschoenen-pumps',
        'damesschoenen-hoge-hakken',
        'platte-schoenen',
        'muiltjes-clogs',
        'damesschoenen-ballerinas',
        'damesschoenen-laarzen',
        'sportschoenen',
        'damesschoenen-slippers',
        'damesschoenen-bruidsschoenen',
        'pantoffels-dames',
        'outdoorschoenen'
    ],
    'damestassen': [
        'tassen-accessoires-handtassen',
        'clutch',
        'tassen-accessoires-shopping-bags',
        'tassen-accessoires-schoudertassen-dames',
        'accessoires-laptop-tassen',
        'tassen-accessoires-sporttassen',
        'heuptassen-dames',
        'tassen-accessoires-rugzakken',
        'luiertassen'
    ],
    'dameszonnebrillen': [
        'tassen-accessoires-zonnebrillen'
    ]
}

HEADER = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"}

MAX_PAGES_PER_CATEGORY = 10

# response = requests.get(ZALANDO_PRODUCT_PAGES['Dames'], headers=HEADER)
   
# soup = BeautifulSoup(response.content, 'html5lib')

# clothing_categories = soup.find('ul', attrs={'class': 'ODGSbs'})

# categories=[]
# for category_element in clothing_categories.descendants:
#     try:
#         category = category_element.get('href').replace('/', '')
#         if category is not None:
#            categories.append(category)
#     except:
#         pass


# Get all product urls
products = []
for product_type, categories in ZALANDO_PRODUCT_PAGES.items():
    for category in categories:
        response = requests.get(f'{ZALANDO_HOME_PAGE}/{category}', headers=HEADER)
        soup = BeautifulSoup(response.content, 'html5lib')
        num_of_pages = int(soup(text=re.compile('Pagina 1 van'))[0].split(' ')[-1])

        # Iterate over all pages in the category
        for page_number in range(1, min(num_of_pages+1, MAX_PAGES_PER_CATEGORY)):
            response = requests.get(f'{ZALANDO_HOME_PAGE}/{category}/?p={page_number}', headers=HEADER)
            soup = BeautifulSoup(response.content, 'html5lib')

            # Get urls to all articles on page
            article_elements = soup.find_all('article')
            for article_element in article_elements:
                try:
                    product_url = article_element.find('a').get('href')
                    if product_url.startswith('https://www.zalando.nl/'):
                        product = {'category': category, 'product_url': product_url}
                        products.append(product)
                    else:
                        print('Product url does not have the right format')
                except:
                    print('Not able to extract the product url from the product element')
        
        print(f'Gathered al product urls for category {category}')


# Remove duplicated product urls
products_clean = []
for product in products:
    product_urls = [product['product_url'] for product in products_clean]
    if not product['product_url'] in product_urls:
        products_clean.append(product)


# Get information and images for each product
try:
    df_products = pd.read_csv('data/products.csv')
    print('Adding to existing products data')
except:
    df_products = pd.DataFrame(columns=['id', 'name', 'url', 'category', 'brand', 'color', 'price'])
    print('Creating new products dataset')


for i, product in enumerate(products_clean):
    # Check if the product data is not downloaded already
    if product['product_url'] not in df_products['url']:
        response = requests.get(product['product_url'], headers=HEADER)
        soup = BeautifulSoup(response.content, 'html5lib')

        # Extract information for the product page
        try:
            brand = soup.find('div', attrs={'class': 'XKeLfu lm1Id5 _2MyPg2'}).text

            name = soup.find('h1', attrs={'class': 'KxHAYs QdlUSH FxZV-M HlZ_Tf wYGQO3 _2MyPg2'}).text

            # Look for regular price
            if soup.find('p', attrs={'class': 'KxHAYs _4sa1cA FxZV-M HlZ_Tf'}) is not None:
                price = soup.find('p', attrs={'class': 'KxHAYs _4sa1cA FxZV-M HlZ_Tf'}).text.split('€\xa0')[-1]
            # Look for discounted price
            elif soup.find('p', attrs={'class': 'KxHAYs _4sa1cA dgII7d Km7l2y'}) is not None:
                price = soup.find('p', attrs={'class': 'KxHAYs _4sa1cA dgII7d Km7l2y'}).text.split('€\xa0')[-1]

            color = soup.find('p', attrs={'class': 'KxHAYs lystZ1 dgII7d HlZ_Tf zN9KaA'}).text

            # Retrieve image urls
            image_urls = []
            images_list_elements = soup.find('ul', attrs={'class': 'XLgdq7 _0xLoFW _78xIQ- _8n7CyI _06gkQU r9BRio xlsKrm _4oK5GO _MmCDa heWLCX KLaowZ'})
            for images_list_element in images_list_elements:
                image_url = images_list_element.find('img')['src']
                image_urls.append(image_url)

            product_id = 0 if df_products.empty else df_products['id'].max() + 1

            df_row = pd.DataFrame({
                                'id': product_id,
                                'name': [name],
                                'url': [product['product_url']],
                                'category': [product['category']],
                                'brand': [brand],
                                'color': [color],  
                                'price': [price],
                                'image_urls': [image_urls]})
            
            df_products = pd.concat([df_products, df_row], ignore_index=True)

        except:
            print(f'Information of product with url {product["product_url"]} could not be extracted')

        # Write output to csv file after each 100 products
        if (i % 100) == 0:
            df_products.to_csv('data/products.csv', index=False)


# Download image for each product in the dataset
for _, row in df_products.iterrows():
    image_id = 1
    product_image_dir = f'data/images/product_{"{:04d}".format(row["id"])}'
    os.mkdir(product_image_dir)
    for image_url in row['image_urls']:

        response = requests.get(image_url)
        if response.status_code == 200:
            with open(f'{product_image_dir}/image_{"{:02d}".format(image_id)}.jpg', 'wb') as file:
                file.write(response.content)

            image_id += 1

        else:
            print(f'Failed to download image with for {image_url}')