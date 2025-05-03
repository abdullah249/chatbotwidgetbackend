from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import chromadb
import os
import pandas as pd
from typing import List, Dict, Any
import shutil
from pathlib import Path
import logging
import sys
from groq import Groq
GROQ_API_KEY="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"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
data_dir = Path("data")
db_dir = Path("db")
data_dir.mkdir(exist_ok=True)
db_dir.mkdir(exist_ok=True)

# Initialize Groq client
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    logger.error(f"Error initializing Groq client: {str(e)}")
    raise

try:
    # Initialize ChromaDB with serverless-friendly configuration
    if os.getenv('VERCEL'):
        # Use in-memory client for serverless environment
        chroma_client = chromadb.Client()
    else:
        # Use persistent client for local development
        chroma_client = chromadb.PersistentClient(path=str(db_dir))
    
    # Create or get the collection
    collection = chroma_client.get_or_create_collection(
        name="knowledge_base"
    )
    logger.info("ChromaDB collection initialized successfully")
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    raise

class Query(BaseModel):
    text: str
    n_results: int = 5
    threshold: float = 0.5  # Minimum similarity threshold

class ChatInput(BaseModel):
    message: str

def process_excel_row(row: pd.Series, columns: List[str]) -> Dict[str, Any]:
    """Process a single row from Excel into a structured document."""
    try:
        # Create a structured text with column names
        text_parts = []
        for col in columns:
            if pd.notna(row[col]):
                text_parts.append(f"{col}: {row[col]}")
        
        # Create metadata
        metadata = {
            "source": "excel",
            "row_id": str(row.name),
            "columns": columns
        }
        
        return {
            "text": "\n".join(text_parts),
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Error processing row {row.name}: {str(e)}")
        return None

@app.post("/api/upload")
async def upload_excel(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are supported")

        # Save the uploaded file
        file_path = data_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Excel file is empty")
        
        # Get column names
        columns = df.columns.tolist()
        
        # Process the Excel data
        documents = []
        for _, row in df.iterrows():
            doc = process_excel_row(row, columns)
            if doc and doc["text"].strip():  # Only add non-empty documents
                documents.append(doc)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid data found in Excel file")
        
        # Add documents to ChromaDB
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        ids = [f"{file.filename}_{i}" for i in range(len(documents))]
        
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully processed {len(documents)} documents from {file.filename}")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Successfully processed {len(documents)} documents from {file.filename}",
                "details": {
                    "total_rows": len(df),
                    "processed_rows": len(documents),
                    "columns": columns
                }
            }
        )
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Excel file is empty or corrupted")
    except Exception as e:
        logger.error(f"Error processing Excel file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_knowledge(query: Query):
    try:
        # First, get relevant context from ChromaDB
        results = collection.query(
            query_texts=[query.text],
            n_results=query.n_results
        )
        
        # Format the context from ChromaDB results
        context = []
        for doc, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            similarity = 1 - distance
            if similarity >= query.threshold:
                context.append({
                    "text": doc,
                    "metadata": metadata,
                    "similarity": similarity
                })
        
        # Prepare the prompt for Groq
        if context:
            context_text = "\n\n".join([f"Context {i+1}:\n{result['text']}" for i, result in enumerate(context)])
            prompt = f"""Based on the following context, please answer the question as your a customer service agent helping them buy a product. If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context:
{context_text}

Question: {query.text}

Answer:"""
        else:
            prompt = f"""I don't have enough information in my knowledge base to answer this question: {query.text}"""

        # Get response from Groq
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. If the answer cannot be found in the context, say so."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=1024,
        )

        response = chat_completion.choices[0].message.content

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "response": response,
                    "context": context if context else []
                }
            }
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    try:
        count = collection.count()
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "document_count": count,
                    "status": "ready"
                }
            }
        )
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(input: ChatInput):
    try:
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
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=1024,
        )

        return {"response": chat_completion.choices[0].message.content}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

# The app will be imported by Vercel's serverless function handler 