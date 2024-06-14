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
        user_data: user data
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    userdata: dict 
    nutritionBranch:str
    diagnosis:str='no'

class Query(BaseModel):
    question: str