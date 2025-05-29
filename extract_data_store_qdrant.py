from playwright.sync_api import sync_playwright
import json
import csv
import time
import hashlib
import pandas as pd
import requests
from PIL import Image
import numpy as np
import torch
import io
from lavis.models import load_model_and_preprocess
from qdrant_client import QdrantClient, models

qdrant_client = QdrantClient(
    url="", 
    api_key="",
)
try:
    qdrant_client.delete_collection("ajio_products")
    print("Deleted existing ajio_products collection.")
except Exception as e:
    print(f"Could not delete ajio_products collection (might not exist): {e}")

try:
    qdrant_client.delete_collection("ajio_embeddings")
    print("Deleted existing ajio_embeddings collection.")
except Exception as e:
    print(f"Could not delete ajio_embeddings collection (might not exist): {e}")

qdrant_client.create_collection(
    collection_name="ajio_products",
    vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE)
)
print("Created ajio_products collection.")

qdrant_client.create_collection(
    collection_name="ajio_embeddings",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
)
print("Created ajio_embeddings collection.")

# --- Create Payload Indexes (Moved outside handle_response - BEST PRACTICE) ---
# These only need to be created once per collection, at the beginning.
try:
    qdrant_client.create_payload_index(
        collection_name="ajio_products",
        field_name="product_type",
        field_schema="keyword"
    )
    print("Created payload index for 'product_type' in ajio_products.")
except Exception as e:
    print(f"Error creating 'type' index: {e}")

try:
    qdrant_client.create_payload_index(
        collection_name="ajio_products",
        field_name="gender",
        field_schema="keyword"
    )
    print("Created payload index for 'gender' in ajio_products.")
except Exception as e:
    print(f"Error creating 'type' index: {e}")

try:
    qdrant_client.create_payload_index(
        collection_name="ajio_products",
        field_name="hashed_url",
        field_schema="keyword"
    )
    print("Created payload index for 'hashed_url' in ajio_products.")
except Exception as e:
    print(f"Error creating 'hashed_url' index in ajio_products: {e}")

try:
    qdrant_client.create_payload_index(
        collection_name="ajio_embeddings",
        field_name="product_type",
        field_schema="keyword"
    )
    print("Created payload index for 'product_type' in ajio_embeddings.")
except Exception as e:
    print(f"Error creating 'type' index: {e}")

try:
    qdrant_client.create_payload_index(
        collection_name="ajio_embeddings",
        field_name="gender",
        field_schema="keyword"
    )
    print("Created payload index for 'gender' in ajio_embeddings.")
except Exception as e:
    print(f"Error creating 'type' index: {e}")
try:
    qdrant_client.create_payload_index(
        collection_name="ajio_embeddings",
        field_name="hashed_url",
        field_schema="keyword"
    )
    print("Created payload index for 'hashed_url' in ajio_embeddings.")
except Exception as e:
    print(f"Error creating 'hashed_url' index in ajio_embeddings: {e}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors= load_model_and_preprocess("blip_feature_extractor", model_type="base", is_eval=True, device=device)

global_product_id_counter = 0

def create_embeddings(text, img):
    processed_text = txt_processors["eval"](text)
    processed_ref = vis_processors["eval"](img).unsqueeze(0).to(device)
        
    with torch.no_grad():
        features = model.extract_features({"image": processed_ref, "text_input": [processed_text]}).multimodal_embeds[:,0,:]
   
    
    query_features = torch.nn.functional.normalize(features, dim=-1)
    
    
    return query_features.cpu().numpy().flatten().tolist()

results = []

def parse_products(data, product_type):

    global global_product_id_counter
    if "products" not in data:
        print("No products found in response.")
        return []

    product_points = []
    embedding_points = []
    for product in data["products"]:
        try:
            alt_text = product.get("images", [{}])[0].get("altText")
            price = product.get("price", {}).get("value")
            mrp = product.get("wasPriceData", {}).get("value")
            url = "https://www.ajio.com" + product.get("url", "")
            gender = product.get("segmentNameText", "")
            image_urls = [img["url"] for img in product.get("images", [])]
            image_url = image_urls[0]

            if mrp and price:
                mrp_value = float(mrp)
                price_value = float(price)
                discount_percent = round(((mrp_value - price_value) / mrp_value) * 100, 2)
            else:
                discount_percent = None
            
            hashed_url = hashlib.md5(url.encode('utf-8')).hexdigest()
            

            payload = {
                "alt_text": alt_text,
                "price": price,
                "mrp": mrp,
                "discount": discount_percent,
                "url": url,
                "hashed_url": hashed_url,
                "image_url": image_url,
                "product_type": product_type,
                "gender": gender
            }

            product_points.append(models.PointStruct(id=global_product_id_counter, vector=[0.0], payload=payload))

            img_response = requests.get(image_url, stream=True)
            image_bytes = io.BytesIO(img_response.content)
            image_from_url = Image.open(image_bytes).convert("RGB")
            embeddings = create_embeddings(text=alt_text, img=image_from_url)

            vector = embeddings
            payload = {
                "alt_text": alt_text,
                "url": url,
                "hashed_url": hashed_url,
                "product_type": product_type,
                "gender": gender
            }
            embedding_points.append(models.PointStruct(id=global_product_id_counter, vector=vector, payload=payload))

            global_product_id_counter+=1

        except Exception as e:
            print("Error parsing product:", e)

    return product_points, embedding_points

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    product_types = ["shirt", "jeans", "kurti", "tshirt", "jacket", "joggars", "top"]
    pages_to_scrape = range(1, 8)

    print("Starting data collection with Playwright...")
    for p_type in product_types:
        print(f"\n--- Collecting data for product type: {p_type.upper()} ---")
        
        all_product_points_for_type = []
        all_embedding_points_for_type = []
        
        for page_num in pages_to_scrape:
            api_url = (
                f"https://www.ajio.com/api/search?"
                f"fields=SITE&currentPage={page_num}&pageSize=500&format=json&"
                f"query={p_type}:relevance&sortBy=relevance&text={p_type}&"
                f"gridColumns=3&advfilter=true&platform=Desktop&is_ads_enable_plp=true&"
                f"is_ads_enable_slp=true&showAdsOnNextPage=true&displayRatings=true&segmentIds="
            )
            print(f"Navigating to {p_type} - Page {page_num}: {api_url}")
            try:
                page.goto(api_url, wait_until="networkidle", timeout=60000)
                # Now the page content is the JSON string. You need to extract it.
                try:
                    json_string = page.text_content('body') # Get content of the body tag
                    json_data = json.loads(json_string) # Parse the string into a Python dict
                    current_product_points, current_embedding_points = parse_products(json_data, p_type)
                except Exception as e:
                    print(f"Error extracting or parsing JSON from page content: {e}")

                all_product_points_for_type.extend(current_product_points)
                all_embedding_points_for_type.extend(current_embedding_points)

                print(f"Processed {len(current_product_points)} products from {p_type} - Page {page_num}. Total products collected for {p_type}: {len(all_embedding_points_for_type)}")

            except Exception as e:
                print(f"Error processing {p_type} - Page {page_num}: {e}")

            time.sleep(2)

        if all_product_points_for_type:
            print(f"\nUpserting {len(all_product_points_for_type)} products for {p_type.upper()} to ajio_products collection...")
            qdrant_client.upsert(collection_name="ajio_products", points=all_product_points_for_type, wait=True)
            print(f"Finished upserting products for {p_type.upper()}.")
        else:
            print(f"No product points to upsert for {p_type.upper()}.")

        if all_embedding_points_for_type:
            print(f"Upserting {len(all_embedding_points_for_type)} embeddings for {p_type.upper()} to ajio_embeddings collection...")
            qdrant_client.upsert(collection_name="ajio_embeddings", points=all_embedding_points_for_type, wait=True)
            print(f"Finished upserting embeddings for {p_type.upper()}.")
        else:
            print(f"No embedding points to upsert for {p_type.upper()}.")

    browser.close()

print("✅ Qdrant collections successfully created and populated with indexes.")