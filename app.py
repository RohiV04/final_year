import streamlit as st
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO

model = SentenceTransformer("clip-ViT-B-32")

qdrant_client = QdrantClient(
    url="https://ade56415-8ec0-4cea-abe7-6ad758da762b.us-east-1-0.aws.cloud.qdrant.io", 
    port=443,
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.aZ2sn0EwwjTi8CtqdcoGKHbIXvTdicEaNsantKRHX-Q",
)

def search_images(query, count):
    """Search for images using either text or image input"""
    results = qdrant_client.search(
        collection_name="images",
        query_vector=model.encode(query).tolist(),
        with_payload=True,
        limit=count
    )
    return results

st.set_page_config(
    page_title="Multi-Modal Image Search Engine",
    layout="wide"
)

st.title("Multi-Modal Image Search Engine")
st.markdown("Semantically search over 15k images using text or image inputs!")

with st.sidebar:
    st.header("Search Controls")
    search_type = st.radio("Search Type", ["Text", "Image"])
    num_results = st.slider("Number of Results", min_value=1, max_value=40, value=8)

# Main content
if search_type == "Text":
    query = st.text_input("Enter your search query", placeholder="Try 'Golden Retriever'")
    search_button = st.button("Search")
    
    if search_button and query:
        results = search_images(query, num_results)
        cols = st.columns(4)
        for idx, result in enumerate(results):
            with cols[idx % 4]:
                try:
                    response = requests.get(result.payload['url'])
                    img = Image.open(BytesIO(response.content))
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to load image {idx + 1}")

else:  # Image search
    uploaded_image = st.file_uploader("Upload an image to search", type=['png', 'jpg', 'jpeg'])
    search_button = st.button("Search")
    
    if search_button and uploaded_image:
        image = Image.open(uploaded_image)
        results = search_images(image, num_results)
        cols = st.columns(4)
        for idx, result in enumerate(results):
            with cols[idx % 4]:
                try:
                    response = requests.get(result.payload['url'])
                    img = Image.open(BytesIO(response.content))
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to load image {idx + 1}")
