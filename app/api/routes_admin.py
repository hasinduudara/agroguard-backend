from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.services.rag_service import process_and_store_pdf

# Create a router object for the admin panel endpoints
router = APIRouter()

@router.post("/upload-guideline", status_code=status.HTTP_201_CREATED)
async def upload_guideline(file: UploadFile = File(...)):
    """
    Admin endpoint to upload agricultural PDF guidelines.
    The PDF is processed, converted to embeddings, and stored in ChromaDB.
    """
    # Validate the file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are allowed."
        )
        
    # Process the PDF and store it in the vector database
    result = await process_and_store_pdf(file)
    return result