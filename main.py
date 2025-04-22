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
    try:
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            # Process a reasonable number of pages for large PDFs
            # This prevents memory issues and timeouts
            max_pages = min(100, len(pdf_reader.pages))
            
            for page_num in range(max_pages):
                try:
                    page_text = pdf_reader.pages[page_num].extract_text()
                    text += page_text + "\n\n"
                except Exception as e:
                    # Skip problematic pages
                    continue
                    
            # Add note if we limited the pages
            if len(pdf_reader.pages) > max_pages:
                text += "\n[Note: This extraction was limited to the first 100 pages due to the document's size.]"
                
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return text


# Function to query Groq API
async def query_groq_api(question, context):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Improved system message for better handling
    system_message = """You are a helpful assistant that answers questions based on the provided PDF content.
    Answer questions only based on the information in the PDF. If the answer isn't in the PDF, say "I couldn't find information about that in the document."
    Keep responses concise and relevant. If the document is very long, focus on the most important parts related to the question."""
    
    # Context length check to prevent token limit issues
    max_context_tokens = 6000  # Conservative estimate for context
    if len(context) > max_context_tokens * 4:  # Rough character to token ratio
        # For very large contexts, summarize first
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                summary_response = await client.post(
                    GROQ_API_URL,
                    json={
                        "model": "llama3-70b-8192",
                        "messages": [
                            {"role": "system", "content": "Create a concise summary of this document, focusing on key information."},
                            {"role": "user", "content": context[:100000]}  # First part for summary
                        ],
                        "temperature": 0.2,
                        "max_tokens": 1000
                    },
                    headers=headers
                )
                summary_response.raise_for_status()
                summary = summary_response.json()["choices"][0]["message"]["content"]
                
                # Use summary + question-specific context
                context = f"DOCUMENT SUMMARY:\n{summary}\n\nRELEVANT SECTIONS:\n{context[:50000]}"
        except:
            # If summarization fails, truncate and proceed
            context = context[:80000] + "\n[Note: The document was truncated due to size limitations.]"
    
    data = {
        "model": "llama3-70b-8192",
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
    except httpx.HTTPStatusError as e:
        error_detail = f"API returned error {e.response.status_code}"
        try:
            error_json = e.response.json()
            if "error" in error_json:
                error_detail += f": {error_json['error']['message']}"
        except:
            pass
        raise HTTPException(status_code=502, detail=error_detail)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error connecting to AI service: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing response: {str(e)}")

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
    
    try:
        # Read file content with size check
        content = await pdf_file.read()
        file_size = len(content)
        
        # Optional: Size limit check (adjust as needed)
        size_limit_mb = 50  # 50MB limit
        if file_size > size_limit_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413, 
                detail=f"PDF file too large. Maximum size is {size_limit_mb}MB. Please optimize your PDF."
            )
        
        # Save the file temporarily
        file_path = f"uploads/{pdf_file.filename}"
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Extract text from the PDF with error handling
        try:
            pdf_text = extract_text_from_pdf(file_path)
            
            if not pdf_text or len(pdf_text.strip()) < 50:
                raise HTTPException(
                    status_code=422, 
                    detail="Could not extract text from the PDF. The file may be encrypted, scanned, or corrupted."
                )
            
            # Store the content in memory
            pdf_content = pdf_text
            current_pdf_name = pdf_file.filename
            
            # Clean up the file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return {"filename": pdf_file.filename, "message": "PDF uploaded successfully"}
        except Exception as e:
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

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
    
    try:
        # Optimize context for very large PDFs to prevent token limit issues
        context = pdf_content
        context_length = len(context)
        
        # If context is extremely large, take a strategic approach
        if context_length > 50000:  # About 12,500 words
            # Split by paragraphs to maintain coherence
            paragraphs = [p for p in context.split("\n\n") if p.strip()]
            
            # Create a reduced context with beginning, relevant sections, and end
            reduced_context = "\n\n".join(paragraphs[:10])  # Beginning
            
            # Simple keyword matching to find relevant sections
            # This is a basic approach - more sophisticated methods could be used
            keywords = question.lower().split()
            relevant_paragraphs = []
            
            for para in paragraphs[10:-5]:
                # Check if paragraph contains any keywords from question
                if any(keyword in para.lower() for keyword in keywords):
                    relevant_paragraphs.append(para)
                    
                # Limit the number of paragraphs to prevent token overflow
                if len(relevant_paragraphs) >= 15:
                    break
                    
            # Add relevant paragraphs and ending
            reduced_context += "\n\n" + "\n\n".join(relevant_paragraphs)
            reduced_context += "\n\n" + "\n\n".join(paragraphs[-5:])  # Ending
            
            # Use the reduced context
            context = reduced_context
        
        # Query Groq API with optimized context
        answer = await query_groq_api(question, context)
        
        return {"answer": answer}
    except httpx.ReadTimeout:
        raise HTTPException(
            status_code=504, 
            detail="The request to the AI service timed out. Please try a more specific question."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


# Main function to run the app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)