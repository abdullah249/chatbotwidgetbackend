from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import logging
import sys
from groq import Groq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chatbot Widget Backend",
    description="Backend API for the Chatbot Widget",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Groq client: {str(e)}")
    raise

class ChatInput(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(input: ChatInput):
    try:
        logger.info(f"Received chat request: {input.message}")
        # Get response from Groq
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions clearly and concisely."
                },
                {
                    "role": "user",
                    "content": input.message
                }
            ],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=1024,
        )

        response = chat_completion.choices[0].message.content
        logger.info("Successfully generated response")
        return JSONResponse(content={"response": response})
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"response": f"Error: {str(e)}"}
        )

@app.get("/api/health")
async def health_check():
    return JSONResponse(content={"status": "healthy"})

@app.get("/")
async def root():
    return JSONResponse(content={"message": "Chatbot Widget Backend is running"})

# Add this for Vercel serverless compatibility
handler = app 