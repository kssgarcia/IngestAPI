#Retrieval grader libraries
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


#Retrieval grader
def retrieval_grader(local_llm:str):
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
    return retrieval_grader