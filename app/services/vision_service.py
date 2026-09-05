import os
import base64
import re
from fastapi import UploadFile, HTTPException, status
from typing import List
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

# Initialize the Groq client
client = Groq(api_key=GROQ_API_KEY)

async def extract_symptoms_from_images(images: List[UploadFile]) -> str:
    """
    Takes a list of uploaded images, encodes them to base64, 
    and sends them to Groq's Qwen Vision model to extract crop symptoms.
    Removes any reasoning/thinking blocks from the output.
    """
    prompt = """
    You are an expert agricultural AI. 
    Analyze the provided images of crops/plants. 
    Identify and describe any visible symptoms of diseases, pests, or nutrient deficiencies.
    Be precise and focus only on the visual symptoms (e.g., 'brown spots on leaves', 'yellowing edges').
    If the plant looks healthy, state that it looks healthy.
    Provide ONLY a brief final diagnosis in 2-4 sentences.
    """
    
    # Initialize the content array with the text prompt
    content = [{"type": "text", "text": prompt}]
    
    try:
        for img in images:
            # Read the image bytes and encode to base64
            img_bytes = await img.read()
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            mime_type = img.content_type or "image/jpeg"
            
            # Construct the data URL required by Groq Vision
            image_url = f"data:{mime_type};base64,{base64_image}"
            
            # Append the image to the content array
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
            
            # Reset file pointer
            await img.seek(0)
            
        # Call the Groq Vision API using the latest supported Qwen model
        # Increased max_completion_tokens to allow the model to finish its thought process
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            model="qwen/qwen3.6-27b",
            temperature=0.2,
            max_completion_tokens=2048 
        )
        
        raw_response = chat_completion.choices[0].message.content
        
        # Clean the response by removing complete <think>...</think> blocks
        clean_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
        
        # Handle cases where the model was cut off and left an unclosed <think> tag
        if '<think>' in clean_response:
            clean_response = re.sub(r'<think>.*', '', clean_response, flags=re.DOTALL).strip()
            
        # If the clean response is completely empty (entire output was inside <think>),
        # grab the last 500 characters of its thought process as a fallback.
        if not clean_response:
            clean_response = raw_response[-500:].strip()
            
        return clean_response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing images with Groq AI: {str(e)}"
        )