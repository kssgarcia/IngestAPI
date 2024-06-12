#Loading documents and store
#document loader libraries
from langchain_community.document_loaders import UnstructuredMarkdownLoader

#splitter libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter

#store libraries
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma

#file path
from .config.config import DATA_DIR 

#Document loader
def document_loader(filename:str):
    filepath=DATA_DIR/filename
    loader = UnstructuredMarkdownLoader(str(filepath))
    data = loader.load()
    return data

#Document splitter
def text_splitter(data:list):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, add_start_index=True
    )
    all_splits = text_splitter.split_documents(data)
    return all_splits

#Store documents and retriever definition
def store_and_retrieve(all_splits:list):
    model_name = "nomic-embed-text-v1.5.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}
    embeddings = GPT4AllEmbeddings(
        model_name=model_name,
        gpt4all_kwargs=gpt4all_kwargs,
        device='cuda'
    )
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
    return vectorstore

def retriever(vectorstore):
    vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})