from typing_extensions import TypedDict
from typing import List
from pydantic import BaseModel

#state flow definition
from langchain.schema import Document

#Graph Creation
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

class Query(BaseModel):
    question: str