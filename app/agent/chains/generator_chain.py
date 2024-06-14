#generatior libraries
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
# from .memoryMethod.summary import summarize_messages

#Generator
def generator(local_llm:str, chat_history):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an nutritionist assistant for question-answering tasks. Use the following pieces of retrieved context to answer the questions your patients have. If you don't know the answer, just say that you don't know. question: {question}\nContext: {context} \nYour asnswers always have into account the data of your patients, allowing them to get a personlized answer based on their objectives, diet and possible deseases, so that they can take actions to change their habits or continue with them in case they are good ones: \ndata:{user_data}\n your answers must be in spanish.",
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

    return generate_with_summarization



#commonGenerator
def commonGenerator(local_llm:str, chat_history):

# Prompt

    promptGeneration = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a kind nutritionist for question-answering tasks. you always have to talk to your patients through messages. answer this patient message: {question}. You must avoid mading things up. whenever you are not sure, just say 'I don't know'.",
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
    rag_common_chain = promptGeneration | llm | StrOutputParser()

    # Run only with message histoy chain
    generate_with_common_message_history = RunnableWithMessageHistory(
        rag_common_chain,
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



    generate_common = (
        RunnablePassthrough.assign(messages_summarized=summarize_messages)
        | generate_with_common_message_history
    )
    
    return generate_common