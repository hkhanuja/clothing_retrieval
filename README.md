# fashion_retrieval

## Idea
The goal of this project is to build a WebApp for multimodal clothing retrieval that makes online fashion search more intuitive and personalized.
Shoppers on platforms like Amazon, Myntra, or Ajio often struggle to express what they're looking for using only keywords or very short texts. This WebApp addresses that by allowing users to:
- Start with a basic search for a clothing item
- Provide a long-form text description to specify the kind of clothing they are looking for
- Optionally select a reference image to further guide the retrieval process

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
