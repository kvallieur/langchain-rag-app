from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
embedding_function = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

def load_and_split_document(file_path: str) -> List[Document]:
    """Loads a document and splits it into chunks. Returns a list of Document objects."""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    documents = loader.load()
    return text_splitter.split_documents(documents)


def upload_doc_to_chroma_vector_db(file_path: str, file_id: int) -> bool:
    """Uploads a document to the Chroma vector database."""
    try:
        splits = load_and_split_document(file_path)
        for split in splits:
            split.metadata = {"file_id": file_id, "filename": os.path.basename(file_path)}
        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error uploading document {file_path}: {e}")
        return False
    

def delete_doc_from_chroma(file_id: int) -> bool:
    """Deletes a document from the Chroma vector database by file_id."""
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")
        
        vectorstore._collection.delete(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id}")
        
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False