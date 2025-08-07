from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field

class ModelName(str, Enum):
    """Allowed LLM model names."""
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"
    GPT4_1_MINI = "gpt-4.1-mini"

class QueryInput(BaseModel):
    """This is the question from the user."""
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)

class QueryResponse(BaseModel):
    """This is the response from the LLM."""
    answer: str
    session_id: str
    model: ModelName

class DocumentInfo(BaseModel):
    """Information about the document."""
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    """Request to delete a file."""
    file_id: int