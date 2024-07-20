#generatior libraries
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOllama
import os
from dotenv import load_dotenv
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

load_dotenv()
MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_URI = f"mongodb+srv://Yilberu:bnnjgIKAm2WzEwd2@cluster0.q4lvimh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"



# # from .memoryMethod.summary import summarize_messages
# summarization_prompt = ChatPromptTemplate.from_messages(
#         [
#             MessagesPlaceholder(variable_name="chat_history"),
#             (
#                 "user",
#                 "Distill the above chat messages into a single summary message. Include as many specific details as you can. keep the names and the data of the user intact.",
#             ),
#         ]
#     )


# #summary chain
# summary_chain = summarization_prompt | llm

#Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an nutritionist assistant for question-answering tasks. Use the following pieces of retrieved context to answer the questions your patients have. If you don't know the answer, just say that you don't know. question: {question}\nContext: {context} \nYour asnswers always have into account the data of your patients, allowing them to get a personlized answer based on their objectives, diet and possible deseases, so that they can take actions to change their habits or continue with them in case they are good ones: \ndata:{user_data}\n your answers must be in spanish.",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{question}")
        
    ]
)



# def get_session_history(session_id):
#     # Usando SQLite como ejemplo de base de datos
#     return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

    # Run with message history and summarisation
# def summarize_messages(sessionid=""):
#         print(sessionid)
#         stored_messages = get_session_history(session_id=sessionid).messages
#         print(stored_messages)
#         if len(stored_messages) <= 8:
#             print("no sumarisation")
#             return False
#         print(len(stored_messages))
        
#         summary_message = summary_chain.invoke({"chat_history": stored_messages})
#         stored_messages.clear()
        
#         stored_messages.add_message(summary_message)

#         return True


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)





#Generator
def generator(local_llm:str, llm:ChatOllama):
    

    # # LLM
    llm.format=None

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    

    # Run only with message histoy chain
    # Run only with message histoy chain
    generate_with_message_history = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGO_URI,
        database_name="snapeatdb",
        collection_name="chatHistories",),
        input_messages_key="question",
        history_messages_key="messages",
    )

    # generate_with_summarization = (
    #     RunnablePassthrough.assign(messages_summarized=summarize_messages)
    #     | generate_with_message_history
    # )

    return generate_with_message_history






# Prompt
promptGeneration = ChatPromptTemplate.from_messages(
    
    [
        (
            "system",
            "You are a kind nutritionist for question-answering tasks. your response should be in json format('answer': 'your answer here'). the user knows what your role is so you don't have to say it in every message, maybe only in first interaction, when the user greets you. You always have to talk to your patients through messages. You must avoid mading things up. whenever you are not sure, just say 'I don't know'. if the name of the patient is available use it always. there are some user data info you know about your patient above. You'll find previous messages below, result of the interactions you had with your patientce, use them to complement your answer in case you need it to answer the patients question. \n this is the question you gotta answer: {question}. \n these are some interactions you had with your patient:\n",
        ),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="memories"),
        ("human", "{question}"),
    ]
)


#commonGenerator
def commonGenerator(local_llm:str, llm:ChatOllama):

    # LLM
    # llm.format=None
    # Chain
    rag_common_chain = promptGeneration | llm.with_config({
            "run_name": "Get Items LLM",
            "tags": ["tool_llm"],  # <-- Propagate callbacks (Python <= 3.10)
        }) | StrOutputParser()

    # #summary chain
    # summary_chain = summarization_prompt | llm

    # Run only with message histoy chain
    generate_with_common_message_history = RunnableWithMessageHistory(
        rag_common_chain,
        lambda session_id: MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGO_URI,
        database_name="snapeatdb",
        collection_name="chatHistories",),
        input_messages_key="question",
        history_messages_key="messages",
    )

    # Run with message history and summarisation
    
    
    return generate_with_common_message_history