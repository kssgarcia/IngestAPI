#lOCAL OLLAMA CHATBOT
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain.schema import Document

#LLM
llm="llama3"

llm= ChatOllama(model=llm, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), format='json', temperature=0)

# Indexing and vectore store


#Loader
markdown_path = "../../IngestAPI/TEMASDENUTRICINBSICABook.md"
loader = UnstructuredMarkdownLoader(markdown_path)

data = loader.load()

#Split

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, add_start_index=True
)
all_splits = text_splitter.split_documents(data)

len(all_splits)

#Store

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma

model_name = "nomic-embed-text-v1.5.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
embeddings = GPT4AllEmbeddings(
    model_name=model_name,
    gpt4all_kwargs=gpt4all_kwargs,
    device='cuda'
)

text = all_splits[80].page_content
print(text)
query_result = embeddings.embed_query(text)
print(query_result)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

# Retriver

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


#Chains

###Retriever grader
llm= ChatOllama(model=local_llm, format='json', temperature=0)

prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation. this an example of the the answer 'score':'yes'""",
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()


###Generate answer chain

# Prompt

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. question: {question}\nContext: {context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
        
    ]
)

# LLM

llm = ChatOllama(model=local_llm, temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run only with message histoy chain
generate_with_message_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="question",
    history_messages_key="messages",
)



# Run with message history and summarisation
def summarize_messages(chain_input):
    stored_messages = chat_history.messages
    if len(stored_messages) <= 12:
        print("no sumarisation")
        return False
    print(len(stored_messages))
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "Distill the above chat messages into a single summary message. Include as many specific details as you can. keep the names and the data of the user intact.",
            ),
        ]
    )
    summarization_chain = summarization_prompt | llm

    summary_message = summarization_chain.invoke({"chat_history": stored_messages})

    chat_history.clear()

    chat_history.add_message(summary_message)

    return True


generate_with_summarization = (
    RunnablePassthrough.assign(messages_summarized=summarize_messages)
    | generate_with_message_history
)
chat_history.clear()


###Question Rewriter

# LLM

llm = ChatOllama(model=local_llm, temperature=0)


# Prompt
re_write_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}.\n Please only return the raw modified version, do not explain what you have done, I only need the modified version. Improved question with no preamble: \n """,
    input_variables=["generation", "question"],
)

question_rewriter = re_write_prompt | llm | StrOutputParser()


###Search tool

from langchain_community.retrievers import TavilySearchAPIRetriever

web_search_tool = TavilySearchAPIRetriever(k=3)


#####Creating graph

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:ß
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation= generate_with_summarization.invoke(
    {"question": question, "context":documents},
    {"configurable": {"session_id": "unused"}},)
    # generation = rag_chain.invoke({"context": documents, "question": question, "messages":chat_history.messages})
    # chat_history.add_ai_message(generation)
    # print(generation)
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        print(d)
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    print (question)

    # Web search
    from langchain_community.retrievers import TavilySearchAPIRetriever

    web_search_tool = TavilySearchAPIRetriever(k=3)
    print("hola")
    docs = web_search_tool.invoke(question)
    web_results = "\n".join([d["content"] for d in docs if "content" in d])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    print("---WEB SEARCH RESULTS---")
    print(documents)

    return {"documents": documents, "question": question}


### Edges


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"