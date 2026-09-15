import os
import shutil
import hashlib
import redis.asyncio as redis
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

# Initialize Async Redis Client
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

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
        # 1. Create a unique cache key based on the exact inputs
        raw_key = f"{user_query}_{ai_symptoms}_{language}"
        cache_key = f"agroadvice_{hashlib.md5(raw_key.encode()).hexdigest()}"
        
        # 2. Check if the answer already exists in Redis Cache
        try:
            cached_response = await redis_client.get(cache_key)
            if cached_response:
                print("\n" + "="*50)
                print("⚡ [REDIS CACHE] Found matching response in cache. Skipping AI processing!")
                print("="*50 + "\n")
                return cached_response
        except Exception as e:
            print(f"⚠️ [REDIS ERROR] Cache check failed (Is Redis running?): {e}")

        # --- CACHE MISS: Proceed with standard AI processing ---
        
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings
        )
        
        search_query = f"{user_query or ''} {ai_symptoms or ''}".strip()
        
        # Search Local Vector Database (PDFs)
        matching_docs = db.similarity_search(search_query, k=3)
        local_context = "\n\n".join([doc.page_content for doc in matching_docs])
        
        print("\n" + "="*50)
        if matching_docs:
            print(f"✅ [VECTOR DB] Found {len(matching_docs)} matching documents in local database.")
        else:
            print("⚠️ [VECTOR DB] No matching documents found in local database.")
        print("="*50 + "\n")
        
        # Search the Internet (Fallback)
        try:
            web_context = web_search.invoke(search_query)
            
            print("\n" + "="*50)
            if web_context and web_context != "No good DuckDuckGo Search Result was found":
                print(f"🌐 [WEB SEARCH] DuckDuckGo search successful. Retrieved {len(web_context)} characters.")
            else:
                print("⚠️ [WEB SEARCH] DuckDuckGo search returned no useful results.")
            print("="*50 + "\n")
            
        except Exception as e:
            web_context = "No web information retrieved."
            print("\n" + "="*50)
            print(f"❌ [WEB SEARCH] Search failed: {str(e)}")
            print("="*50 + "\n")

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
            1. First, determine if your final answer relies mostly on the 'Official Database Context' or the 'Internet Search Context'.
            2. Start your response with EXACTLY ONE of these lines to show the source:
               - **මූලාශ්‍රය: අපගේ දත්ත ගබඩාව** (If you used the local database)
               - **මූලාශ්‍රය: අන්තර්ජාලය** (If you used the internet search)
            3. VERY IMPORTANT: Sinhala text consumes a huge amount of AI tokens. To prevent the response from cutting off, you MUST keep your answer EXTREMELY SHORT and concise.
            4. Use only 2 to 4 very brief bullet points. DO NOT generate large tables or long paragraphs. 
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
        final_answer = response.content

        # 3. Save the newly generated answer to Redis for 24 hours (86400 seconds)
        try:
            await redis_client.set(cache_key, final_answer, ex=86400)
            print("✅ [REDIS CACHE] New response successfully saved to cache.")
        except Exception as e:
            print(f"⚠️ [REDIS ERROR] Could not save response to cache: {e}")

        return final_answer
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate crop advice: {str(e)}"
        )