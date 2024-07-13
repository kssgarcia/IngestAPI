from .memory_setup import chroma_database_setUP
from IngestAPI.app.agent.chains.memory_creator import memo_creator
from summary import summary_chain
documents_vectorestore, memory_collection, chat_collection, memory_chat_collection = chroma_database_setUP()
from datetime import datetime
import uuid
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

local_llm="llama3"

memo_creator=memo_creator(local_llm=local_llm)
summary_chain=summary_chain(local_llm=local_llm)

def add_memory(messages: list):
    document = [message.content for message in messages]
    ids = [str(uuid.uuid4()) for _ in range(len(messages))]
    memory_chat_collection.add(documents=document, ids=ids)

def add_messages(messages: list,sesion_id: str):
    document = []
    for message in messages:
        if isinstance(message, HumanMessage):
            document.append("paciente: " + message.content)
        elif isinstance(message, AIMessage):
            document.append("nutricionista: " + message.content)
        else:
            document.append(message.content)  # En caso de que el tipo no sea reconocido
    ids = [str(uuid.uuid4()) for _ in range(len(messages))]
    chat_collection.add(documents=document, ids=ids, metadatas={"sesion_id": sesion_id})
    

# Eliminar todos los documentos en la colección
def deleteChat():
    chat_collection.delete(ids=chat_collection.get()['ids'])

def deletememory():
    memory_collection.delete(ids=memory_collection.get()['ids'])

def deletesummaries():
    memory_chat_collection.delete(ids=memory_chat_collection.get()['ids'])


def add_lil_memo(query:str):

    lil_memo=memo_creator.invoke(input={"statement":query})
    memories=lil_memo['memories']
    # Generar un UUID único para el problem_id
    problem_id = str(uuid.uuid4())

    # Agregar metadatos con timestamps actuales y problem_id para cada memoria
    metadatas = [{"problem_id": problem_id, "timestamp": datetime.now().isoformat()} for _ in memories]

    # Generar UUIDs únicos para cada memoria
    ids = [str(uuid.uuid4()) for _ in memories]

    # Agregar las memorias a la base de datos vectorial
    memory_collection.add(
        documents=memories,
        metadatas=metadatas,
        ids=ids
    )

def get_lil_memo(question):
    resultados_iniciales = memory_collection.query(query_texts=question,
    n_results=5  # Obtenemos múltiples resultados para considerar recencia
    )
    
    if resultados_iniciales["documents"][0]:

        # Filtrar y priorizar resultados por recencia
        resultados_filtrados = sorted(
        resultados_iniciales["documents"][0],
        key=lambda x: datetime.fromisoformat(resultados_iniciales["metadatas"][0][resultados_iniciales["documents"][0].index(x)]["timestamp"]),
        reverse=True
        )

        # Obtener el problem_id del resultado más reciente
        memoria_relevante = resultados_filtrados[0]
        problem_id_relevante = resultados_iniciales["metadatas"][0][resultados_iniciales["documents"][0].index(memoria_relevante)]["problem_id"]

        # Paso 4: Recuperar todas las memorias relacionadas usando el problem_id
        resultados_relacionados = memory_collection.query(query_texts=[""],
            where={"problem_id": problem_id_relevante}
        )

        # Extraer las memorias relacionadas
        memorias_relacionadas = sorted(
            resultados_relacionados["documents"][0],
            key=lambda x: datetime.fromisoformat(resultados_relacionados["metadatas"][0][resultados_relacionados["documents"][0].index(x)]["timestamp"]),
            reverse=True
        )

        # Resultado final
        respuesta = {
            "problema expresado por el paciente": memoria_relevante,
            "memorias de paciente": memorias_relacionadas
        }

        return respuesta
    else:
        return resultados_iniciales["documents"][0]
    

def summary(chat:list):
    content=summary_chain.invoke({"chat_history":chat})
    summary=AIMessage(content=content)
    add_memory([summary])

def get_memories(question:str):
    memory_results = memory_chat_collection.query(query_texts=question, n_results=1)["documents"][0]
    lil_memo=get_lil_memo(question=question)
    if lil_memo:
        lil_memo= get_lil_memo(question=question)["memorias de paciente"]
    if memory_results:
        lil_memo.append(memory_results)
    lil_memo=[AIMessage(content=content) for content in lil_memo]
    return lil_memo
get_memories("hay algo que me ayude a reducir mi peso?")  