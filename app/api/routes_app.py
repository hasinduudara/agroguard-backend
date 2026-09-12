from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from typing import List, Optional

from app.services.vision_service import extract_symptoms_from_images
from app.services.rag_service import get_crop_advice

router = APIRouter()

@router.post("/analyze-crop")
async def analyze_crop(
    text_query: Optional[str] = Form(None),
    language: str = Form("si"), # "si" for Sinhala, "en" for English
    images: Optional[List[UploadFile]] = File(None)
):
    image_names = []
    ai_symptoms = None

    if images:
        if len(images) > 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Maximum 3 images are allowed."
            )
        image_names = [img.filename for img in images if img.filename]
        
        try:
            ai_symptoms = await extract_symptoms_from_images(images)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AI Processing failed: {str(e)}"
            )
    
    if not text_query and not image_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide at least a text query or an image."
        )

    try:
        # Pass the selected language to the RAG service
        final_advice = await get_crop_advice(
            user_query=text_query, 
            ai_symptoms=ai_symptoms,
            language=language
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve advice from knowledge base: {str(e)}"
        )

    return {
        "status": "success",
        "message": "Crop analyzed successfully.",
        "received_text": text_query,
        "extracted_symptoms": ai_symptoms,
        "final_advice": final_advice,
        "language_used": language,
        "image_count": len(image_names)
    }