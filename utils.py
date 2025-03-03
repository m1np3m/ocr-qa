from pydantic import BaseModel
from typing import Optional, Literal
import json
import base64
from openai import OpenAI

client = OpenAI()


class IdentityCard(BaseModel):
    """Data model for a user's identity card."""

    full_name: Optional[str]
    sex: Literal["Nam", "Nữ", "Khác"]
    address: Optional[str]
    date_of_birth: Optional[str]


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_user_info(image_path):
    if isinstance(image_path, str):
        base64_image = encode_image(image_path)
    else:
        base64_image = image_path
    # Getting the Base64 string

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
    Extract the main information in the image and return it in JSON format without special characters.
    The fields of the data follow the underscore rule. If the document language is in Vietnamese, pay more attention to the Vietnamese characters.                    
    """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        response_format=IdentityCard,
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content) 
