import os
from fastapi import UploadFile, HTTPException, status
from typing import List
from dotenv import load_dotenv

# Import the new google-genai package
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Configure the Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

# Initialize the new Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

async def extract_symptoms_from_images(images: List[UploadFile]) -> str:
    """
    Takes a list of uploaded images, sends them to Gemini Vision using the new SDK,
    and returns a textual description of the crop symptoms.
    """
    prompt = """
    You are an expert agricultural AI. 
    Analyze the provided images of crops/plants. 
    Identify and describe any visible symptoms of diseases, pests, or nutrient deficiencies.
    Be precise and focus only on the visual symptoms (e.g., 'brown spots on leaves', 'yellowing edges').
    If the plant looks healthy, state that it looks healthy.
    """
    
    # Add the text prompt to the contents list first
    contents = [prompt]
    
    try:
        # Read each uploaded image and format it for the new Gemini SDK
        for img in images:
            img_bytes = await img.read()
            contents.append(
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type=img.content_type or "image/jpeg",
                )
            )
            # Reset the file pointer so it can be read again if needed
            await img.seek(0)
            
        # Use the recommended Chat API with the latest gemini-3.6-flash model
        chat = client.chats.create(model="gemini-3.6-flash")
        response = chat.send_message(contents)
        
        return response.text
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing images with AI: {str(e)}"
        )