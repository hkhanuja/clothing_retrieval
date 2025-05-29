# Clothing Retrieval

## Idea
The goal of this project is to build a WebApp for multimodal clothing retrieval that makes online fashion search more intuitive and personalized.
Shoppers on platforms like Amazon, Myntra, or Ajio often struggle to express what they're looking for using only keywords or very short texts. This WebApp addresses that by allowing users to:
- Start with a basic search for a clothing item
- Provide a long-form text description to specify the kind of clothing they are looking for
- Optionally select a reference image to further guide the retrieval process

# Flow of the project
- First I scraped data from Ajio (https://www.ajio.com/) which is one of India's leading F&L brand. I use playwright to extract data from Ajio API for different clothing items.
- Since my idea involved dealing with multimodal retreival I decided to use vector database instead of traditional relational databases. I created an account on QDrant to host my data.
- I extracted relevant information about each clothing item, and create BLIP (Vision-Language Model) embeddings using the description text and image of the item.
- I create two tables _ajio_products_ and _ajio_embeddings_ with relevant metadata (called payload in QDrant) to make filtering easier (like gender and clothing_type).
- My backend is written using FastAPI interacts with the hosted QDrant database to retrieve the top 20 items closest to user query and optinally selected reference link (from which image is extracted).
- Currently I am working on the Frontend which is being written in Reactjs and the project will be hosted on AWS EC2.

# Reproducibility

1. **Clone the repository**
   ```
   git clone https://github.com/hkhanuja/fashion_retrieval.git
   cd your-repo
   ```
2. **Create conda environment and install dependencies**
   ```
   conda create -n env_name python=3.10
   conda activate env_name
   pip install -r requirements.txt
   ```

3. **Create account on QDrant and get your credentials**
   ```
   python extract_data_store_qdrant.py
   ```
   
3. **Run backend server**
   ```
   cd fast_api
   uvicorn main:app --reload
   ```
