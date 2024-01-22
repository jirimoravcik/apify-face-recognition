from apify_client import ApifyClient
from dotenv import load_dotenv
import os
import requests
import io
from unidecode import unidecode
from PIL import Image
from insightface.app import FaceAnalysis
import numpy as np

load_dotenv()
IDENTITY_VECTORS_FOLDER_NAME = os.environ.get("IDENTITY_VECTORS_FOLDER_NAME")
APIFY_TOKEN = os.environ.get("APIFY_TOKEN")

app = FaceAnalysis(
    name="buffalo_l", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0)

if __name__ == "__main__":
    client = ApifyClient(APIFY_TOKEN)

    os.makedirs(IDENTITY_VECTORS_FOLDER_NAME, exist_ok=True)

    run_info = client.actor("jirimoravcik/apify-about-page-scraper").call()
    dataset = client.dataset(dataset_id=run_info["defaultDatasetId"])
    for person in dataset.iterate_items():
        img_raw = requests.get(person["image_url"]).content
        img = Image.open(io.BytesIO(img_raw)).convert("RGB")
        file_name = unidecode(person["name"].replace(" ", "_"))
        # img.save(f'./data/{filename}.jpg')
        faces = app.get(np.array(img))
        if len(faces) == 0:
            print(
                f"No face found for {file_name}. Consider checking the original image."
            )
            continue
        embedding = faces[0].normed_embedding
        np.save(
            os.path.join(IDENTITY_VECTORS_FOLDER_NAME, file_name + ".npy"),
            embedding,
        )
