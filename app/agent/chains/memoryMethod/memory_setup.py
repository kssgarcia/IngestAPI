#chroma

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma
import chromadb



import chromadb.utils.embedding_functions as embedding_functions

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.document_loaders import UnstructuredMarkdownLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter



def chroma_database_setUP():
    #Embedding function
    embedding_function= SentenceTransformerEmbeddings(model_name="intfloat/multilingual-e5-large", model_kwargs = {'device': 'cuda'})

    # client_documents=chromadb.PersistentClient(path="./document_chroma_db")
    client_memory=chromadb.PersistentClient(path="./memories_chroma_db")
    client_chat= chromadb.PersistentClient(path="./chat_chroma_db")

    chat_collection=chat_collection=client_chat.get_collection(name="user_chat2")
    memory_collection=client_memory.get_collection(name="memories2")
    memory_chat_collection=client_memory.get_collection(name="five_memories")

    # retriever_memory=vectorstore_memories.as_re
    documents_vectorestore=Chroma(
        persist_directory="./document_chroma_db",
        collection_name="documents",
        embedding_function=embedding_function,
    )

    # momories_vectorestore = Chroma(
    #     client=client_memory,
    #     collection_name="memories2",
    #     embedding_function=embedding_function,
    # )
    # chat_vectorestore = Chroma(
    #     client=client_chat,
    #     collection_name="user_chat2",
    #     embedding_function=embedding_function,
    # )

    # memory_chat_collection= Chroma(
    #     client=memory_chat_collection,
    #     collection_name="five_memories",
    #     embedding_function=embedding_function
    # )

    return (documents_vectorestore, memory_collection, chat_collection, memory_chat_collection)


