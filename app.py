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
import hashlib
from huggingface_hub import InferenceClient

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

# Initialize Hugging Face Inference client
hf_client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
)

def image_with_stable_diffusion(prompt):
    """Generate image using Stable Diffusion via Hugging Face Inference API"""
    try:
        # Generate image using the Inference API
        image = hf_client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-3.5-large"
        )
        return image
    except Exception as e:
        st.error(f"Error generating image with Stable Diffusion: {str(e)}")
        return None

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
        # collection_name="midjourney",
        collection_name="images",
        query_vector=query_vector.tolist(),
        with_payload=True,
        limit=count
    )
    print(f"Search results: {results}")
    return results

def store_image_in_qdrant(image_url, prompt):
    """Store image in Qdrant with its embeddings"""
    try:
        # Download and encode the image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image_vector = encode_image(image)
        
        # Create a unique ID for the image
        image_id = int(hashlib.md5(image_url.encode()).hexdigest()[:8], 16)
        
        # Store in Qdrant
        qdrant_client.upsert(
            # collection_name="midjourney",
            collection_name="images",
            points=[
                {
                    "id": image_id,
                    "vector": image_vector.tolist(),
                    "payload": {
                        "url": image_url,
                        "prompt": prompt,
                        "type": "generated"
                    }
                }
            ]
        )
        return True
    except Exception as e:
        st.error(f"Error storing image in Qdrant: {str(e)}")
        return False

def generate_image(prompt):
    """Generate image using DALL-E and store in Qdrant"""
    try:
        result = azure_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1
        )
        image_url = json.loads(result.model_dump_json())['data'][0]['url']
        
        # Store the generated image in Qdrant
        if store_image_in_qdrant(image_url, prompt):
            return image_url
        return None
    except Exception as e:
        print(e)
        st.error(f"Error generating image Try again later")
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
    /* Main theme colors */
    :root {
        --primary-color: #6C5CE7;
        --secondary-color: #A8A4E3;
        --background-color: #F8F9FA;
        --text-color: #2D3436;
        --accent-color: #81ECEC;
    }

    /* Global styles */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: white;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        background-color: white;
        border-radius: 0.5rem;
        gap: 0.5rem;
        padding: 1rem;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--accent-color);
        color: var(--text-color);
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }

    /* Container styling */
    .content-container {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }

    /* Input styling */
    .stTextInput input, .stTextArea textarea {
        background-color: #F1F3F4;
        border: 1px solid transparent;
        border-radius: 0.5rem;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px var(--secondary-color);
    }

    /* Button styling */
    .stButton button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background-color: var(--secondary-color);
        transform: translateY(-2px);
    }

    /* Radio button styling */
    .stRadio [data-testid="stMarkdownContainer"] > div {
        background-color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        display: flex;
        gap: 1rem;
    }

    /* Slider styling */
    .stSlider [data-testid="stThumbValue"] {
        background-color: var(--primary-color);
        color: white;
    }

    /* Success/Error message styling */
    .stSuccess, .stError {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }

    /* Image container styling */
    .stImage {
        border-radius: 0.5rem;
        overflow: hidden;
        transition: transform 0.3s ease;
    }

    .stImage:hover {
        transform: scale(1.02);
    }

    /* Footer styling */
    footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
        text-align: center;
        color: var(--text-color);
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
    st.markdown("Search through our image and text database using text or upload your own image!")
    
    search_type = st.radio("Search Type", ["Text", "Image"], horizontal=True)
    
    with st.container():
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
                                # response = requests.get(result.payload['image_url'])
                                img = Image.open(BytesIO(response.content))
                                st.image(img, use_container_width=True)
                            except Exception as e:
                                st.error(f"Failed to load image {idx + 1}")
        st.markdown('</div>', unsafe_allow_html=True)

# Image Generation Tab
with tabs[1]:
    st.header("Image Generation")
    st.markdown("Generate unique images!")
    
    with st.container():
        prompt = st.text_area("Enter your image prompt", 
                            placeholder="A serene landscape with mountains reflecting in a crystal clear lake at sunset",
                            height=100)
        generate_button = st.button("🎨 Generate Image")
        
        if generate_button and prompt:
            with st.spinner("Generating your image..."):
                # image = image_with_stable_diffusion(prompt)
                # if image:
                #     st.success("Image generated successfully!")
                #     st.image(image, caption="Generated Image", use_container_width=True)
                image_url = generate_image(prompt)
                if image_url:
                    st.success("Image generated successfully!")
                    st.image(image_url, caption="Generated Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
# st.markdown('<footer>Built with ❤️ using OpenAI, CLIP, and Streamlit</footer>', unsafe_allow_html=True)
