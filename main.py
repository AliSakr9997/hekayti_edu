from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from google.auth import default
from google.auth.transport.requests import Request

app = FastAPI()

PROJECT_ID = "hekayti-education"
LOCATION = "us-central1"
MODEL = "imagen-4.0-generate-001"

VERTEX_ENDPOINT = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
    f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL}:predict"
)


class ImageRequest(BaseModel):
    prompt: str


def get_access_token():
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    return credentials.token


@app.post("/generate-image")
def generate_image(data: ImageRequest):
    try:
        access_token = get_access_token()

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "instances": [
                {
                    "prompt": data.prompt
                }
            ]
        }

        response = requests.post(
            VERTEX_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=response.text
            )

        result = response.json()
        image_base64 = result["predictions"][0]["bytesBase64Encoded"]

        return {
            "image_base64": image_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
