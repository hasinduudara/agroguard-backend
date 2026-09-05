import os
import shutil
from fastapi import UploadFile, HTTPException, status
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Ensure API keys are available
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

# Initialize Google Embeddings (Using the embedding-001 model for ChromaDB)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Initialize the Groq Chat model for generating the final answer
# Using the latest supported fast model (llama-3.1-8b-instant) to avoid deprecation issues
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
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

async def get_crop_advice(user_query: str, ai_symptoms: str = None) -> str:
    """
    Searches the ChromaDB vector store for relevant agricultural data 
    based on the user's query and the symptoms identified by the vision AI.
    Generates a final answer using Groq LLM.
    """
    try:
        # 1. Load the existing vector database
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings
        )
        
        # 2. Search for the top 3 most relevant document chunks
        search_query = f"{user_query or ''} {ai_symptoms or ''}".strip()
        matching_docs = db.similarity_search(search_query, k=3)
        
        # Combine the retrieved texts
        context = "\n\n".join([doc.page_content for doc in matching_docs])
        
        # 3. Define the prompt template for the AI
        prompt_template = PromptTemplate(
            input_variables=["context", "symptoms", "query"],
            template="""
            You are a highly knowledgeable Agricultural Advisor. 
            Use the following context extracted from official agricultural documents to answer the user's question.
            
            Context from documents:
            {context}
            
            Observed Plant Symptoms (from Image AI):
            {symptoms}
            
            User's Query:
            {query}
            
            Instructions:
            1. Analyze the symptoms and the query based ONLY on the provided context.
            2. Identify the possible disease/issue and recommend specific treatments or fertilizers mentioned in the context.
            3. If the context does not contain the answer, clearly state that you do not have enough information based on the official guidelines, but provide general safe advice if possible.
            4. Keep the answer structured and easy to read.
            """
        )
        
        # 4. Generate the final response
        final_prompt = prompt_template.format(
            context=context,
            symptoms=ai_symptoms if ai_symptoms else "None provided",
            query=user_query if user_query else "What is the issue with this crop and how to treat it?"
        )
        
        response = llm.invoke(final_prompt)
        return response.content
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate crop advice: {str(e)}"
        )