import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from google.auth import default
from google.auth.transport.requests import Request

app = FastAPI()

# Configuration
PROJECT_ID = "hekayti-education"
LOCATION = "us-central1"
# imagen-3.0 is the current stable version for most projects
MODEL = "imagen-3.0-generate-001" 

VERTEX_ENDPOINT = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
    f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL}:predict"
)

class ImageRequest(BaseModel):
    prompt: str

def get_access_token():
    """Acquires Application Default Credentials"""
    try:
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        return credentials.token
    except Exception as e:
        print(f"Auth Error: {e}")
        return None

@app.get("/")
def health_check():
    """Critical for Cloud Run to verify the container started"""
    return {"status": "alive"}

@app.post("/generate-image")
def generate_image(data: ImageRequest):
    access_token = get_access_token()
    if not access_token:
        raise HTTPException(status_code=500, detail="Authentication failed")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "instances": [{"prompt": data.prompt}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "1:1"
        }
    }

    try:
        response = requests.post(
            VERTEX_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=90 # Image generation can take time
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=response.text)

        result = response.json()
        # Ensure the AI didn't filter the image
        prediction = result["predictions"][0]
        if "bytesBase64Encoded" not in prediction:
            return {"error": "Image was filtered by Safety Filters", "reason": prediction.get("raiFilteredReason")}

        return {"image_base64": prediction["bytesBase64Encoded"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
