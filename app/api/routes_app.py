from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from typing import List, Optional

# Import the vision_service here
from app.services.vision_service import extract_symptoms_from_images

# Create a router object for the mobile app endpoints
router = APIRouter()

@router.post("/analyze-crop")
async def analyze_crop(
    text_query: Optional[str] = Form(None),
    images: Optional[List[UploadFile]] = File(None)
):
    """
    Endpoint to receive up to 3 images and an optional text query.
    Extracts symptoms using Gemini Vision AI if images are provided.
    """
    
    image_names = []
    ai_symptoms = None

    # 1. Analyze images if they are provided
    if images:
        if len(images) > 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Maximum 3 images are allowed."
            )
        image_names = [img.filename for img in images if img.filename]
        
        # Extract symptoms from images using Gemini Vision API
        try:
            ai_symptoms = await extract_symptoms_from_images(images)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AI Processing failed: {str(e)}"
            )
    
    # 2. Send an error if neither text nor image is provided
    if not text_query and not image_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide at least a text query or an image."
        )

    # 3. Send the response with the AI result
    return {
        "status": "success",
        "message": "Crop analyzed successfully.",
        "received_text": text_query,
        "extracted_symptoms": ai_symptoms,  # Symptoms provided by Gemini AI
        "image_count": len(image_names)
    }