"""This is the main FastAPI application file for the LangChain RAG Agent."""
import os
import uuid
import logging
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from db_utils import (
    get_all_documents,
    insert_application_logs,
    get_chat_history,
    insert_document_record,
    delete_document_record
)
from chroma_utils import (
    upload_doc_to_chroma_vector_db,
    delete_doc_from_chroma
)
from langchain_utils import get_rag_chain

# Configure logging with proper format and handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()  # Also log to console
    ]
)

# Create logger for this module
logger = logging.getLogger(__name__)


app = FastAPI()

# @app.on_event("startup")
# async def startup_event():
#     """Initialize the application on startup."""
#     # Create temp directory if it doesn't exist
#     os.makedirs("./temp", exist_ok=True)
#     logger.info("Application started successfully")

@app.post("/chat", response_model=QueryResponse)
def chat(input: QueryInput):
    """Handles user queries and returns responses from the LLM."""
    logger.info(
        "Received query: %s with session_id: %s using model: %s", 
        input.question, 
        input.session_id, 
        input.model
    )
    session_id = input.session_id or str(uuid.uuid4())

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(model=input.model.value)
    
    response = rag_chain.invoke({
        "input": input.question,
        "chat_history": chat_history
    })
    
    logger.info(
        "Generated response: %s for session_id: %s", 
        response['answer'], 
        session_id
    )
    insert_application_logs(
        session_id, 
        input.question, 
        response['answer'], 
        input.model.value
    )
    return QueryResponse(
        answer=response['answer'], 
        session_id=session_id, 
        model=input.model
    )


@app.post("/upload-doc")
def upload_and_index_doc(file: UploadFile = File(...)):
    """Uploads a document and indexes it in the Chroma vector database."""
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {file_extension}. "
                f"Allowed types are: {', '.join(allowed_extensions)}"
            )
        )
    
    temp_file_path = f"./temp/{file.filename}"

    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        file_id = insert_document_record(file.filename)
        logger.info(
            "Inserted document record with file_id: %s for file: %s", 
            file_id, 
            file.filename
        )

        embedded_succesfully = upload_doc_to_chroma_vector_db(
            temp_file_path, 
            file_id
        )

        if embedded_succesfully:
            logger.info(
                "Successfully uploaded and embedded document: %s with file_id: %s", 
                file.filename, 
                file_id
            )
            return {
                "message": "Document uploaded and embedded successfully", 
                "file_id": file_id
            }
        else:
            delete_document_record(file_id)
            raise HTTPException(
                status_code=500, 
                detail="Failed to embed the document in the vector database"
            )
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info("Removed temporary file: %s", temp_file_path)

@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    """Lists all documents stored in the vector database."""
    documents = get_all_documents()
    logger.info("Retrieved %d documents from the database", len(documents))
    return documents


@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    """Deletes a document from the vector database and the local storage."""
    # Delete from Chroma
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        # If successfully deleted from Chroma, delete from our database
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {
                "message": (
                    f"Successfully deleted document with file_id "
                    f"{request.file_id} from the system."
                )
            }
        else:
            return {
                "error": (
                    f"Deleted from Chroma but failed to delete document "
                    f"with file_id {request.file_id} from the database."
                )
            }
    else:
        return {
            "error": (
                f"Failed to delete document with file_id "
                f"{request.file_id} from Chroma."
            )
        }
