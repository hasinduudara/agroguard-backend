import os
import jwt
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware # Added CORS import
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize the FastAPI application
app = FastAPI(title="AgroGuard AI API", version="1.0.0")

# Add CORS middleware to allow the React frontend to communicate with this backend API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins (e.g., http://localhost:5173)
    allow_credentials=True,
    allow_methods=["*"], # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# Setup HTTP Bearer to extract the token from the Authorization header
security = HTTPBearer()

# Retrieve the Supabase JWT secret from environment variables
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

def verify_supabase_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dependency function to verify the Supabase JWT token.
    It extracts the token from the request, decodes it using the Supabase JWT secret,
    and returns the user payload if valid.
    """
    token = credentials.credentials
    
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT Secret is not configured on the server."
        )

    try:
        # Decode the token. 
        # Supabase uses the HS256 algorithm and "authenticated" audience by default.
        payload = jwt.decode(
            token, 
            SUPABASE_JWT_SECRET, 
            algorithms=["HS256"], 
            audience="authenticated"
        )
        return payload
        
    except jwt.ExpiredSignatureError:
        # Handle expired tokens
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
        )
    except jwt.InvalidTokenError:
        # Handle invalid or malformed tokens
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )

# ==========================================
# API Routes
# ==========================================

from app.api.routes_app import router as app_router
from app.api.routes_admin import router as admin_router

app.include_router(app_router, prefix="/api/app", tags=["Mobile App"])
app.include_router(admin_router, prefix="/api/admin", tags=["Admin Panel"])

@app.get("/")
def root_endpoint():
    """
    Public health check endpoint to verify the server is running.
    """
    return {"message": "Welcome to AgroGuard AI API"}

@app.get("/api/test-auth")
def test_authentication(user_data: dict = Depends(verify_supabase_token)):
    """
    A protected test route.
    This will only return a success message if a valid Supabase JWT token is provided.
    """
    # The 'sub' claim in the JWT payload contains the user's unique ID (UUID)
    user_id = user_data.get("sub")
    
    return {
        "message": "Authentication successful!", 
        "user_id": user_id
    }