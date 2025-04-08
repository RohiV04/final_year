import streamlit as st
import os
from qdrant_client import QdrantClient
import open_clip
import torch
from PIL import Image
import requests
from io import BytesIO
from openai import AzureOpenAI
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Initialize clients
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    port=443,
    api_key=os.getenv("QDRANT_API_KEY"),
)

azure_client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

def encode_text(text):
    """Encode text input using CLIP"""
    with torch.no_grad():
        text_tokens = tokenizer(text)
        text_features = model.encode_text(text_tokens.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]

def encode_image(image):
    """Encode image input using CLIP"""
    with torch.no_grad():
        image_input = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0]

def search_images(query, count):
    """Search for images using either text or image input"""
    if isinstance(query, str):
        query_vector = encode_text(query)
    else:
        query_vector = encode_image(query)
        
    results = qdrant_client.search(
        collection_name="images",
        query_vector=query_vector.tolist(),
        with_payload=True,
        limit=count
    )
    return results

def generate_image(prompt):
    """Generate image using DALL-E"""
    try:
        result = azure_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1
        )
        return json.loads(result.model_dump_json())['data'][0]['url']
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

# Page configuration
st.set_page_config(
    page_title="AI Image Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stTabs {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stTab {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        margin: 5px;
    }
    .search-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # st.image("https://raw.githubusercontent.com/openai/openai-python/main/logo.png", width=100)
    st.title("AI Image Platform")
    st.markdown("---")
    num_results = st.slider("Number of Results", min_value=1, max_value=40, value=8)

# Main content
tabs = st.tabs(["🔍 Image Search", "🎨 Image Generation"])

# Image Search Tab
with tabs[0]:
    st.header("Multi-Modal Image Search")
    st.markdown("Search through our image database using text or upload your own image!")
    
    search_type = st.radio("Search Type", ["Text", "Image"], horizontal=True)
    
    with st.container():
        # st.markdown('<div class="search-container">', unsafe_allow_html=True)
        if search_type == "Text":
            query = st.text_input("Enter your search query", placeholder="Try 'Golden Retriever'")
            search_button = st.button("🔍 Search", key="text_search")
            
            if search_button and query:
                with st.spinner("Searching..."):
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
            search_button = st.button("🔍 Search", key="image_search")
            
            if search_button and uploaded_image:
                with st.spinner("Searching..."):
                    image = Image.open(uploaded_image)
                    st.image(image, caption="Uploaded Image", width=200)
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
        st.markdown('</div>', unsafe_allow_html=True)

# Image Generation Tab
with tabs[1]:
    st.header("DALL-E Image Generation")
    st.markdown("Generate unique images using OpenAI's DALL-E model!")
    
    with st.container():
        # st.markdown('<div class="search-container">', unsafe_allow_html=True)
        prompt = st.text_area("Enter your image prompt", 
                            placeholder="A serene landscape with mountains reflecting in a crystal clear lake at sunset",
                            height=100)
        generate_button = st.button("🎨 Generate Image")
        
        if generate_button and prompt:
            with st.spinner("Generating your image..."):
                image_url = generate_image(prompt)
                if image_url:
                    st.success("Image generated successfully!")
                    st.image(image_url, caption="Generated Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using OpenAI, CLIP, and Streamlit")
