from typing import Union
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient, models
import ast
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from fastapi import Request
import requests
import io

from fastapi import FastAPI

app = FastAPI()


@app.on_event("startup")
def load_initial_information():
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors= load_model_and_preprocess("blip_feature_extractor", model_type="base", is_eval=True, device=device)

    qdrant_client = QdrantClient(
        url="https://f1b3703b-9c3e-47a4-97ea-ab612bee7106.us-east4-0.gcp.cloud.qdrant.io:6333", 
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.GuUKx_7JRai4ui6KGQPK8w_Vp-OL1uPR0OKXjB0pTM8",
    )

    app.state.device = device
    app.state.model = model
    app.state.vis_processors = vis_processors
    app.state.txt_processors = txt_processors
    app.state.qdrant_client = qdrant_client

        
@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running. Qdrant is ready."}


@app.get("/items/{item_id}")
async def request_item_list(request: Request, item_id: str, gender: str, max_items: Union[int, None] = None):
    if max_items is None:
        max_items = 100
    qdrant_client = request.app.state.qdrant_client
    res = qdrant_client.scroll(
            collection_name="ajio_products",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="product_type",
                        match=models.MatchValue(value=item_id),
                    ),
                    models.FieldCondition(
                        key="gender",
                        match=models.MatchValue(value=gender),
                    )
                ]
            ),
            limit = max_items,
            with_payload = True
        )
    return {"item_id": item_id, "max_items": max_items, "res": res}

@app.get("/search_items/{item_id}")
async def request_item(request: Request, item_id: str, gender: str, ref_text: str, ref_image_hash: Union[str, None] = None):
    model = request.app.state.model
    txt_processors = request.app.state.txt_processors
    vis_processors = request.app.state.vis_processors
    qdrant_client = request.app.state.qdrant_client
    device =request.app.state.device

    
    processed_text = txt_processors["eval"](ref_text)

    if ref_image_hash:
        res = qdrant_client.scroll(
            collection_name="ajio_products",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="product_type",
                        match=models.MatchValue(value=item_id),
                    ),
                    models.FieldCondition(
                        key="hashed_url",
                        match=models.MatchValue(value=ref_image_hash),
                    ),
                    models.FieldCondition(
                        key="gender",
                        match=models.MatchValue(value=gender),
                    )
                ]
            ),
        )



        image_url = res[0][0].payload["image_url"]
        img_response = requests.get(image_url, stream=True)
        image_bytes = io.BytesIO(img_response.content)
        image_from_url = Image.open(image_bytes)
        processed_ref = vis_processors["eval"](image_from_url).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model.extract_features({"image": processed_ref, "text_input": [processed_text]}).multimodal_embeds[:,0,:]
    else:
        with torch.no_grad():
            features = model.extract_features({ "text_input": [processed_text]}, mode="text").text_embeds[:,0,:]

    query_features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy().flatten().tolist()
    # results = qdrant_client.query_points(
    #     collection_name="ajio_embeddings",
    #     query=models.Query(
    #         vector=query_features,
    #         filter=models.Filter(
    #             must=[
    #                 models.FieldCondition(
    #                     key="gender",
    #                     match=models.MatchValue(value=gender),
    #                 )
    #             ]
    #         )
    #     ),
    #     limit=5,
    #     with_payload=True
    # )
    results = qdrant_client.search(
        collection_name="ajio_embeddings",
        query_vector=query_features,
        query_filter=models.Filter(   # <-- Use 'query_filter' for the search method
            must=[
                models.FieldCondition(
                    key="gender",
                    match=models.MatchValue(value=gender), # Filter for gender 'Men'
                )
            ]
        ),
        limit=20, # Number of results to return
        with_payload=True # Include payload in results so you can see the gender
    )


    return {"test":results}