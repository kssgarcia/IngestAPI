#Rewriter libraries
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

#Rewriter
def rewriter(llm:ChatOllama):
    re_write_prompt = PromptTemplate(
        template="""You a question re-writer that converts an input question to a better version that is optimized. The optimized one should keep the same languge \n 
        for vectorstore retrieval. Look at the initial and formulate an improved question. \n
        Here is the initial question: \n\n {question}.\n Please only return the raw modified version, do not explain what you have done, I only need the modified version. Improved question with no preamble: \n """,
        input_variables=["generation", "question"],
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter