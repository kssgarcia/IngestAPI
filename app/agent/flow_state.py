#state flow definition
from langchain.schema import Document
from .chains.generator_chain import generator
from .chains.retriever_grader_chain import retrieval_grader
from .chains.rewriter_chain import rewriter
from .retriever import *

#state
from .graph_state import GraphState, Query

#web tool library
# Web search
from langchain_community.retrievers import TavilySearchAPIRetriever

#memory libraries
from langchain.memory import ChatMessageHistory


local_llm="llama3"



#Document loader
docuemt=document_loader(filename="TEMASDENUTRICINBSICABook.md")
#splitter
all_splits=text_splitter(docuemt)
#Embedding and vector store
vectorstore = store_and_retrieve(all_splits)

#memory
chat_history = ChatMessageHistory()
 
#tools
web_search_tool = TavilySearchAPIRetriever(k=3)
retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#Chains
generate_with_summarization=generator(local_llm=local_llm,chat_history=chat_history)
retrieval_grader=retrieval_grader(local_llm=local_llm)
question_rewriter=rewriter(local_llm=local_llm)




# #Graph Creation
# class GraphState(TypedDict):
#     """
#     Represents the state of our graph.

#     Attributes:
#         question: question
#         generation: LLM generation
#         web_search: whether to add search
#         documents: list of documents
#     """

#     question: str
#     generation: str
#     web_search: str
#     documents: List[str]


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
    print(question, documents)

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
    web_documents = state["documents"]
    print (question)

    web_search_tool = TavilySearchAPIRetriever(k=3)
    docs = web_search_tool.invoke(str(question))
    web_results = "\n".join([d["content"] for d in docs if "content" in d])
    web_results = Document(page_content=web_results)
    web_documents.append(web_results)

    print("---WEB SEARCH RESULTS---")
    print(web_documents)

    return {"documents": web_documents, "question": question}


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