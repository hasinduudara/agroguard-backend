from fastapi import APIRouter, File, UploadFile, HTTPException, status, Depends
from app.main import verify_supabase_token
from app.services.rag_service import process_and_store_pdf

# Create a router object for the admin panel endpoints
router = APIRouter()

# Change the endpoint path to match the React frontend URL
@router.post("/upload-pdf", status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    file: UploadFile = File(...),
    user_data: dict = Depends(verify_supabase_token)
):
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