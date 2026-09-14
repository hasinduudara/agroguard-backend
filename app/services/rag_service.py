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
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

# Initialize Google Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Initialize the Groq Chat model
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5, 
    max_tokens=800 
)

# Initialize the Web Search Tool
web_search = DuckDuckGoSearchRun()

CHROMA_PATH = "chroma_db"

async def process_and_store_pdf(file: UploadFile):
    temp_file_path = f"temp_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
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
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

async def get_crop_advice(user_query: str, ai_symptoms: str = None, language: str = "si") -> str:
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings
        )
        
        search_query = f"{user_query or ''} {ai_symptoms or ''}".strip()
        
        # 1. Search Local Vector Database (PDFs)
        matching_docs = db.similarity_search(search_query, k=3)
        local_context = "\n\n".join([doc.page_content for doc in matching_docs])
        
        # 2. Search the Internet (Fallback/Augmentation)
        try:
            web_context = web_search.invoke(search_query)
        except Exception:
            web_context = "No web information retrieved."

        # Combine both contexts
        combined_context = f"--- Official Database Context ---\n{local_context}\n\n--- Internet Search Context ---\n{web_context}"
        
        lang_instruction = (
            "5. IMPORTANT: You MUST write the final response entirely in Sinhala language (using Sinhala script, not English)."
            if language == "si" 
            else "5. IMPORTANT: You MUST write the final response entirely in English."
        )
        
        prompt_template = PromptTemplate(
            input_variables=["context", "symptoms", "query", "lang_instruction"],
            template="""
            You are a highly knowledgeable Agricultural Advisor helping Sri Lankan farmers. 
            Use the following context (which includes both local database info and recent internet data) to answer the user's question.
            
            Context:
            {context}
            
            Observed Plant Symptoms (from Image AI):
            {symptoms}
            
            User's Query:
            {query}
            
            Instructions:
            1. Analyze the symptoms and the query based ONLY on the provided context.
            2. Identify the possible disease/issue and recommend specific treatments or fertilizers mentioned in the context.
            3. Prioritize the 'Official Database Context' if there are conflicts. Use the 'Internet Search Context' to fill in gaps.
            4. Keep the answer structured, concise, and easy to read using Markdown tables or lists.
            5. CRITICAL: DO NOT repeat the same words or phrases endlessly. Write natural, fluent, and meaningful sentences.
            {lang_instruction}
            """
        )
        
        final_prompt = prompt_template.format(
            context=combined_context,
            symptoms=ai_symptoms if ai_symptoms else "None provided",
            query=user_query if user_query else "What is the issue with this crop and how to treat it?",
            lang_instruction=lang_instruction
        )
        
        response = llm.invoke(final_prompt)
        return response.content
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate crop advice: {str(e)}"
        )