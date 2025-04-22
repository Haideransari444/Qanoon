from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Dict, Optional, Union
import os
import re
import asyncio
import logging
from dotenv import load_dotenv
# Import LangChain modules
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
VECTOR_STORE_PATH = r"E:\legeslative bot\vectordb"
DOCUMENTS_PATH = r"E:\legeslative bot\data"  # Path to your documents folder

# Load embeddings model
logger.info("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector store
logger.info("Loading vector store...")
vectorstore = FAISS.load_local(
    VECTOR_STORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Initialize LLM
logger.info("Initializing language model...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
)

# Modified prompt template to remove sources from answer body
template = """
You are a Pakistani legal expert assistant. Use only the provided legal documents to answer the user's question.
If the documents do not contain relevant information, answer using your knowledge of Pakistani law.
Legal Documents:
{context}
User's Question:
{question}
Your response must follow these rules:
    Provide a clear, explainatory, and fact-based answer using appropriate legal terminology.
    Do not reference the Quran or Sunnah in your answer.
    Base your response strictly on the provided legal documents or authentic Pakistani law knowledge.
    DO NOT include any document citations, references, or sources within the body of your answer.
    Do not infer or fabricate information not explicitly present in the legal documents.
    Exclude any documents that are unrelated to the question.
    Do not include a "Sources:" section in your answer.
    Your answer should be complete and stand alone without any reference to sources.
"""
PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Create the chain
def format_docs(docs):
    return "\n\n".join(f"Document: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}\nContent: {doc.page_content}" for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# Document cache
document_cache = {}

# Create FastAPI app
app = FastAPI(title="Pakistani Legal Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Request models
class QuestionRequest(BaseModel):
    question: str
    
    @validator('question')
    def question_must_be_valid(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty or just whitespace')
        if len(v) < 5:
            raise ValueError('Question must be at least 5 characters long')
        if len(v) > 500:
            raise ValueError('Question must be less than 500 characters long')
        return v

class SourceResponse(BaseModel):
    source_id: str
    display_name: str
    content: Optional[str] = None  # Now includes the document content

class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]

# Function to extract relevant document IDs from retrieved documents
def extract_document_ids_from_retrieved(docs):
    """Extract document IDs from retrieved documents"""
    sources = []
    for doc in docs:
        source = doc.metadata.get('source', '')
        if source:
            # Extract just the filename from the source path
            filename = os.path.basename(source)
            if filename and filename not in sources:
                sources.append(filename)
    return sources

def get_document_text(source_id: str) -> str:
    """Get the text content of a document by its source ID"""
    # Validate source_id to prevent path traversal
    if not re.match(r'^[A-Za-z0-9_\-\.]+\.(txt|pdf)$', source_id):
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    
    # Check cache first
    if source_id in document_cache:
        return document_cache[source_id]
    
    # Construct file path
    file_path = os.path.join(DOCUMENTS_PATH, source_id)
    
    # Try to find the file with case-insensitive matching if not found directly
    if not os.path.exists(file_path):
        found = False
        try:
            for filename in os.listdir(DOCUMENTS_PATH):
                if filename.lower() == source_id.lower():
                    file_path = os.path.join(DOCUMENTS_PATH, filename)
                    found = True
                    break
        except Exception as e:
            logger.error(f"Error searching for document: {str(e)}")
        
        if not found:
            raise HTTPException(status_code=404, detail=f"Document not found: {source_id}")
    
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Cache the content
        document_cache[source_id] = content
        return content
    except Exception as e:
        logger.error(f"Error reading document {source_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading document: {str(e)}")

async def get_document_text_async(source_id: str) -> str:
    """Async version to get the text content of a document by its source ID"""
    try:
        # Run synchronous file operations in a thread pool
        return await asyncio.to_thread(get_document_text, source_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in async document retrieval for {source_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

# API endpoints
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Process a legal question and return the answer with sources and their content"""
    try:
        logger.info(f"Processing question: {request.question}")
        
        # Get relevant documents using the retriever
        docs = retriever.invoke(request.question)
        
        # Extract source IDs from retrieved documents
        source_ids = extract_document_ids_from_retrieved(docs)
        logger.info(f"Retrieved {len(source_ids)} relevant sources")
        
        # Get answer from LLM
        raw_answer = qa_chain.invoke(request.question)
        logger.info("Got response from LLM")
        
        # Process source documents
        sources = []
        for source_id in source_ids:
            try:
                sources.append(SourceResponse(
                    source_id=source_id,
                    display_name=f"Judgment: {source_id}",
                    # Don't include content here to keep response size smaller
                    content=None
                ))
            except Exception as e:
                logger.error(f"Error processing source {source_id}: {str(e)}")
        
        # Create response
        response = AnswerResponse(
            answer=raw_answer,
            sources=sources
        )
        
        logger.info(f"Returning response with {len(response.sources)} sources")
        return response
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/document/{source_id}")
async def get_document(source_id: str):
    """Return the content of a document by its source ID"""
    try:
        logger.info(f"Retrieving document: {source_id}")
        content = await get_document_text_async(source_id)
        return {"content": content, "source_id": source_id}
    except HTTPException as e:
        logger.error(f"HTTP error retrieving document: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error retrieving document {source_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Also verify that documents path exists and is accessible
    docs_exist = os.path.exists(DOCUMENTS_PATH) and os.path.isdir(DOCUMENTS_PATH)
    docs_readable = os.access(DOCUMENTS_PATH, os.R_OK) if docs_exist else False
    
    return {
        "status": "ok", 
        "vector_store_path": VECTOR_STORE_PATH,
        "documents_path": DOCUMENTS_PATH,
        "documents_exist": docs_exist,
        "documents_readable": docs_readable
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Pakistani Legal Assistant API")
    uvicorn.run(app, host="127.0.0.1", port=8000)