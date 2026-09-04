import os
import shutil
from fastapi import UploadFile, HTTPException, status
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure GEMINI_API_KEY is available
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

# Initialize Google Embeddings (Using the embedding-001 model)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Define the directory where ChromaDB will store the vector data locally
CHROMA_PATH = "chroma_db"

async def process_and_store_pdf(file: UploadFile):
    """
    Saves the uploaded PDF temporarily, extracts text, chunks it, 
    and stores the vector embeddings in ChromaDB.
    """
    temp_file_path = f"temp_{file.filename}"
    
    try:
        # 1. Save the uploaded file temporarily to the local disk
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Load and parse the PDF document using LangChain
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        # 3. Split the text into smaller chunks for accurate retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # 4. Convert chunks to embeddings and store them in ChromaDB
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        
        return {
            "status": "success",
            "message": f"Successfully processed {file.filename}", 
            "chunks_created": len(chunks)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing the PDF document: {str(e)}"
        )
    finally:
        # 5. Clean up the temporary file from the server
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)