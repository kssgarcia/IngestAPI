### Memory
# from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama


summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                """Distill the above chat messages into a single summary message. Include as many specific details as you can. keep the names and the data of the user intact. The idea is that you focus the resume in three core parts: the problem, a pattern in case one exist and the solution provided. this is an example of a conversation: \n 
                Paciente (P): Hola, doctora. Estoy pensando en comerme una porción de lasaña esta noche, pero no estoy seguro de cómo afectará mi cuerpo. ¿Es una buena idea?
                Nutricionista (N): ¡Hola! La lasaña es deliciosa, pero también es importante considerar su impacto en tu salud. ¿Qué ingredientes tiene la lasaña que planeas comer?
                P: Tiene carne molida, queso, salsa de tomate y pasta.
                N: Bien. La carne molida proporciona proteínas y hierro, pero también puede ser alta en grasas saturadas. El queso aporta calcio, pero también puede ser alto en grasas. La salsa de tomate es rica en licopeno, un antioxidante, pero a veces contiene azúcares añadidos. Y la pasta es una fuente de carbohidratos.
                P: ¿Y cómo puedo minimizar los efectos negativos?
                N: Puedes:
                - Controlar las porciones: No exageres con la cantidad.
                - Equilibrar con vegetales: Acompaña la lasaña con una ensalada o verduras al vapor.
                - Elegir opciones más saludables: Usa carne magra o incluso sustituye la carne por espinacas o berenjenas.
                - Limitar el queso: Opta por quesos más bajos en grasa.
                - Estar atento a las salsas: Busca opciones de salsa de tomate sin azúcares añadidos.
                P: ¡Gracias! Lo tendré en cuenta. ¿Y si ya la he comido?
                N: Si ya la has comido, no te preocupes. Haz una caminata o realiza alguna actividad física para ayudar a procesar los nutrientes. Y recuerda, una comida no define tu salud en general. ¡Disfruta tu lasaña con moderación!
                P: Entendido. ¡Gracias por la orientación!
                N: ¡De nada! Estoy aquí para ayudarte.\n

                and this is an example of a resume: \n
                Resumen de memorias recientes:
                - Paciente de nombre (en caso de que haya un nombre) pregunta frecuentemente sobre el impacto de ciertos alimentos en su salud.
                - Nutricionista proporciona recomendaciones específicas para controlar porciones, equilibrar con vegetales, y elegir opciones más saludables.
                Patrones observados: Paciente necesita orientación continua sobre cómo hacer elecciones alimenticias saludables.
                Recomendaciones: Continuar ofreciendo consejos específicos y prácticos, y revisar el progreso en las próximas sesiones.

                don't forget to keep details as names and preferences intact.
                """,
            ),
        ]
    )


def summary_chain(local_llm:str, llm:ChatOllama, summarization_prompt:ChatPromptTemplate=summarization_prompt, ):
    #llm
    #summary chain
    summary_chain = summarization_prompt | llm | StrOutputParser()
    return summary_chain

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