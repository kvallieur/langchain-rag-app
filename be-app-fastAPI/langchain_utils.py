"""This module contains utility functions for LangChain-based RAG (Retrieval-Augmented Generation) systems."""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from chroma_utils import vectorstore
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# This retriever retrieves the top 2 documents based on the user's query
plain_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

SYSTEM_MESSAGE_TO_CONTEXTUALIZE_QUERY = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE_TO_CONTEXTUALIZE_QUERY),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# FOr the actual query

QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the provided context."),
    ("system", "Context: {context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

def get_rag_chain(model="gpt-4.1-mini"):
    """Creates a Retrieval-Augmented Generation (RAG) chain."""
    llm = ChatOpenAI(model=model, temperature=0.0)
    history_aware_retriever = create_history_aware_retriever(
        llm,
        plain_retriever,
        CONTEXTUALIZE_Q_PROMPT
    )                                                         
    question_answer_chain = create_stuff_documents_chain(llm, QUERY_PROMPT)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain
