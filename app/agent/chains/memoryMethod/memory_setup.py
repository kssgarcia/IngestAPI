#chroma

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma
import chromadb

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

#Embedding function
embedding_function= SentenceTransformerEmbeddings(model_name="intfloat/multilingual-e5-large")

def chroma_database_setUP():

    
    # client_documents=chromadb.PersistentClient(path="./document_chroma_db")
    client_memory=chromadb.PersistentClient(path="./memories_chroma_db")
    client_chat= chromadb.PersistentClient(path="./chat_chroma_db")

    chat_collection=chat_collection=client_chat.get_or_create_collection(name="user_chat")
    memory_collection=client_memory.get_or_create_collection(name="memories")
    memory_chat_collection=client_memory.get_or_create_collection(name="five_memories")

    return (memory_collection, chat_collection, memory_chat_collection)

def document_vector_store():
    # retriever_memory=vectorstore_memories.as_re
    documents_vectorestore=Chroma(
        persist_directory="./document_chroma_db",
        collection_name="documents",
        embedding_function=embedding_function,
    )

    return documents_vectorestore


