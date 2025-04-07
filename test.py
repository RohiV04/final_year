from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
qdrant_client = QdrantClient(
    url="https://ade56415-8ec0-4cea-abe7-6ad758da762b.us-east-1-0.aws.cloud.qdrant.io", 
    port=443,
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.aZ2sn0EwwjTi8CtqdcoGKHbIXvTdicEaNsantKRHX-Q",
)

# qdrant_client.create_collection(
#    collection_name="images",
#    vectors_config = rest.VectorParams(size=512, distance = rest.Distance.COSINE),
# )
from sentence_transformers import SentenceTransformer
from PIL import Image
# Load the model
model = SentenceTransformer("clip-ViT-B-32")
import pandas as pd
data = pd.read_csv('./data/images.tsv', sep='\t', header=None).reset_index()
print(data.shape, data.head(), sep="\n")
import urllib
import os

def download_file(url):
    os.makedirs("./images", exist_ok=True)
    basename = os.path.basename(url)
    target_path = os.path.join("./images", basename)
    if not os.path.exists(target_path):
        try:
            urllib.request.urlretrieve(url, target_path)
        except urllib.error.HTTPError:
            return None
    return target_path
print(qdrant_client.get_collections())
def upsert_to_db(points):
  qdrant_client.upsert(
   collection_name="images",
   points=[
      rest.PointStruct(
            id=point['id'],
            vector=point['vector'].tolist(),
            payload=point['payload']
      )
      for point in points
   ]
)
points = []

for i in range(0,500):
  img = download_file(data.iloc[i,1])
  if (img):
    embedding = model.encode(Image.open(str(img)))
    points.append({
        "id":i,
        "vector":embedding,
        "payload":{"url":data.iloc[i,1]}
    })
  if((i+1)%50 == 0):
    print(f"{i+1} images encoded")

upsert_to_db(points)
print("\nEmbeddings upserted to vector database.")

