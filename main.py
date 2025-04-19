from dotenv import load_dotenv
load_dotenv()
# Import required libraries
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import PyPDF2
import httpx
import uvicorn
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="PDF Question Answering Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for storing uploaded files
os.makedirs("uploads", exist_ok=True)

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store PDF content in memory
pdf_content = {}
current_pdf_name = None

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to query Groq API
async def query_groq_api(question, context):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create a system message that instructs the model to answer based on the PDF context
    system_message = """You are a helpful assistant that answers questions based on the provided PDF content. 
    Answer questions only based on the information in the PDF. If the answer isn't in the PDF, say "I couldn't find information about that in the document." 
    Keep responses concise and relevant."""
    
    data = {
        "model": "llama3-70b-8192",  # Using Llama 3 70B model
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"PDF content: {context}\n\nQuestion: {question}"}
        ],
        "temperature": 0.2,
        "max_tokens": 800
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(GROQ_API_URL, json=data, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Error querying Groq API: {str(e)}")

# Root endpoint - returns HTML interface
@app.get("/", response_class=HTMLResponse)
async def index():
    # This will serve the HTML from the frontend file
    with open("templates/index.html", "r") as f:
        return f.read()

# Endpoint to upload PDF file
@app.post("/upload/")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    global pdf_content, current_pdf_name
    
    # Validate file type
    if not pdf_file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save the file temporarily
    file_path = f"uploads/{pdf_file.filename}"
    with open(file_path, "wb") as f:
        content = await pdf_file.read()
        f.write(content)
    
    # Extract text from the PDF
    try:
        pdf_text = extract_text_from_pdf(file_path)
        
        # Store the content in memory
        pdf_content = pdf_text
        current_pdf_name = pdf_file.filename
        
        # Clean up the file
        os.remove(file_path)
        
        return {"filename": pdf_file.filename, "message": "PDF uploaded successfully"}
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# Endpoint to check current PDF
@app.get("/current-pdf/")
async def get_current_pdf():
    return {"pdf_name": current_pdf_name}

# Endpoint to ask questions about the PDF
@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    global pdf_content
    
    if not pdf_content:
        raise HTTPException(status_code=400, detail="No PDF has been uploaded yet")
    
    # Query Groq API with the question and PDF content
    answer = await query_groq_api(question, pdf_content)
    
    return {"answer": answer}

# Main function to run the app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)