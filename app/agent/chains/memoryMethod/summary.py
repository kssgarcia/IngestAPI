### Memory
# from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat_history = ChatMessageHistory()


#summarize messages
# Run with message history and summarisation
def summarize_messages(chain_input, chat_history):
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