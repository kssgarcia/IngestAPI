#state flow definition
from langchain.schema import Document
from .chains.generator_chain import *
from .chains.retriever_grader_chain import retrieval_grader
from .chains.rewriter_chain import rewriter
from .chains.analyser import analyser
from .chains.brancher import branchDecider
from .retriever import *
from .chains.planner import plannerchain
from .chains.memory_decider import memo_decider
from .chains.memory_creator import memo_creator 
from .chains.memoryMethod.summary import summary_chain
from .chains.memory_management import summary, add_lil_memo, get_memories
from .chains.memoryMethod.memory_setup import document_vector_store
from langchain_core.runnables import RunnableConfig

#mongo config

# Obtener la cadena de conexión desde una variable de entorno
MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")


# Obtener la cadena de conexión desde una variable de entorno
mongo_uri = f"mongodb+srv://Yilberu:bnnjgIKAm2WzEwd2@cluster0.q4lvimh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

#state
from .graph_state import GraphState, Query

#web tool library
# Web search
from langchain_community.retrievers import TavilySearchAPIRetriever

#memory libraries
from langchain_community.chat_message_histories import ChatMessageHistory

#planner
import re
# Regex to match expressions of the form E#... = ...[...]
regex_pattern = r"Plan:\s*(.+?)\s*#(E\d+)\s*=\s*(\w+)\[(.+?)\]"


solve_prompt = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might
contain irrelevant information.
\n
{plan}
\n
Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.
\n
Task: {task}
Response:"""

local_llm="llama3.1"
local_llm2="phi3:mini"
# llm = ChatOllama(model=local_llm, temperature=0, num_ctx=10000)
def model() -> ChatOllama:
    llm_json= ChatOllama(model=local_llm, temperature=0, format="json", num_ctx=10000)
    return llm_json
print("************************************iNICIADO MODELO****************")

llm_json=model()
# #Document loader
# docuemt=document_loader(filename="TEMASDENUTRICINBSICABook.md")
# #splitter
# all_splits=text_splitter(docuemt)
# #Embedding and vector store
# vectorstore = store_and_retrieve(all_splits)




#tools
web_search_tool = TavilySearchAPIRetriever(k=3)
retriever=document_vector_store()
retriever=retriever.as_retriever(search_type="mmr", search_kwargs={"k": 3})

#Chains
generator=generator(local_llm=local_llm, llm=llm_json)
retrieval_grader=retrieval_grader(local_llm=local_llm, llm_json=llm_json)
# question_rewriter=rewriter(local_llm=local_llm)
analyser=analyser(local_llm=local_llm, llm=llm_json)

#analyser
def set_analyser(analyser=analyser):
    return analyser

branchDecider=branchDecider(local_llm=local_llm, llm_json=llm_json)
commonGenerator=commonGenerator(local_llm=local_llm,llm=llm_json)
planner=plannerchain(local_llm=local_llm,llm=llm_json)

summary_chain=summary_chain(local_llm=local_llm, llm=llm_json)
memo_creator=memo_creator(local_llm=local_llm, llm_json=llm_json)
memo_decider=memo_decider(local_llm=local_llm, llm_json=llm_json)


def inicialize(state):
    question=state["question"]
    sessionid= state["sessionid"]
    user_data= state["user_data"]
    nutritionRequired=state["nutritionBranch"]
    documents=state["documents"]
    """     
    Initializes the state of the document
    """
    print("----------INITIALIZE")
    return {"question":question, "sessionid":sessionid, "user_data":user_data, "nutritionBranch":nutritionRequired, "documents":documents}

async def branch(state):
    """
    Decides wether documents are needed or not based on the question

    Args:
    state (Document): The current state of the document

    Returns:
    state (Document): The new state with a new nuutritionalBranch key
    """
    print("---BRANCH---")
    question=state["question"]
    diagnosis = state["diagnosis"]
    sessionid= state["sessionid"]
    diagnosis=state["diagnosis"]
    user_data=state["user_data"]
    required = await branchDecider.ainvoke({"sentence":question})
    return {"question":question, "nutritionBranch":required, "sessionid":sessionid, "diagnosis":diagnosis, "user_data":user_data}

async def nutritionRequired(state):
    """
    Decides whether the given question requires nutritional information

    Args:
        state (dict): The current graph state

    Returns:ß
        Binary decision, generate or go into the nutritional branch
    """
    print("---Nutrition Required---")
    required=state["nutritionBranch"]


    # Retrieval
    
    if required["nutrition"]=='no':
        print("---NO NUTRITION REQUIRED---")
        return "generateCommon"
    
    if required["nutrition"]=='yes':
        print("---NUTRITION REQUIRED---")
        return "retrieve"

async def generateCommon(state):
    """
    Generate common answers to the question

    Args:
    state (dict): The current graph state

    Returns:
    state(dict): The updated graph state with a new generation key
    """
    question = state["question"]
    sessionid= state["sessionid"]
    user_data=state["user_data"]
    # messages = []

    chat_message_history = MongoDBChatMessageHistory(
    session_id=sessionid,
    connection_string=mongo_uri,
    database_name="snapeatdb",
    collection_name="chatHistories",
    )
    print(chat_message_history.messages[-1:-3])
    
    # if required["nutrition"]=='no':
    memories=get_memories(question=question)
    memories.append(user_data)
    print("---GENERATE COMMON---")
    messages=await commonGenerator.ainvoke(input={"question":question,"messages":chat_message_history.messages[-1:-3], "memories":memories}, config={"configurable": {"session_id": sessionid}})
        # messages.append(message.content)
    print(messages)
    return { "generation":[AIMessage(content="".join(messages))]}




async def retrieve(state):
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
    # if nutritionRequired["nutrition"]=='yes':
    documents = await retriever.ainvoke(question)
    return {"documents": documents}



# async def formatuserdata(state):
#     """ 
#     Analyze user data and format it for it to be used in the model.

#     Args:
#      state (dict): The current graph state
    
#     Returns:
#         state (dict): New key added to state, userdata, that contains formatted user data
#     """
#     print("---FORMATUSERDATA---")
#     user_data = state["userdata"]
#     nutritionRequired=state["nutritionBranch"]

#     # Format user data
#     formatted_data = (
#         f"name:{user_data.name},"
#         f"lastName:{user_data.lastName},"
#         f"Email: {user_data.email}, "
#         f"Age: {user_data.dietetics.age}, "
#         f"Lifestyle: {user_data.dietetics.lifestyle}, "
#         f"Job: {user_data.dietetics.job}, "
#         f"Anthropometry: (Height: {user_data.dietetics.anthropometry.height}, "
#         f"Weight: {user_data.dietetics.anthropometry.weight}, "
#         f"BMI: {user_data.dietetics.anthropometry.BMI}, "
#         f"Waist Circumference: {user_data.dietetics.anthropometry.waist_circumference}), "
#         f"Biochemical Indicators: (Glucose: {user_data.dietetics.biochemical_indicators.glucose}, "
#         f"Cholesterol: {user_data.dietetics.biochemical_indicators.cholesterol}), "
#         f"Diet: (Ingest Preferences: {', '.join(user_data.dietetics.diet.ingest_preferences)}, "
#         f"Fruits and Vegetables: {user_data.dietetics.diet.fruits_and_vegetables}, "
#         f"Fiber: {user_data.dietetics.diet.fiber}, "
#         f"Saturated Fats: {user_data.dietetics.diet.saturated_fats}, "
#         f"Sugars: {user_data.dietetics.diet.sugars}, "
#         f"Today Meals: {', '.join(user_data.dietetics.diet.today_meals)}), "
#         f"Social Indicators: (Marital Status: {user_data.dietetics.social_indicators.marital_status}, "
#         f"Income: {user_data.dietetics.social_indicators.income}, "
#         f"Access to Healthy Foods: {user_data.dietetics.social_indicators.access_to_healthy_foods}), "
#         f"Physical Activity: {user_data.dietetics.physical_activity}, "
#         f"Daily Activities: {', '.join(user_data.dietetics.daily_activities)}, "
#         f"Activities Energy Consumption: {user_data.dietetics.activities_energy_consumption}, "
#         f"Goals: (Reduce Weight: {user_data.dietetics.goals}, "
#         f"Potential Diseases: (Type 2 Diabetes: {user_data.dietetics.potential_diseases}, "
#     )

#     messages=[]

#     if nutritionRequired["nutrition"]=='yes':
#         async for message in analyser.astream(input={formatted_data}):
#             messages.append(message)
#         return {"userdata": user_data, "diagnosis":[AIMessage(content=" ".join(messages))]}
#     else:
#         return {"diagnosis":[AIMessage(content="")]}



async def generate(state):
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
    sessionid= state["sessionid"]
    user_data=state["user_data"]
    web_search=state["web_search"]
    nutritionRequired=state["nutritionBranch"]
    result=state["result"]
    
    # RAG generation
    if web_search=="no" and nutritionRequired["nutrition"]=='yes':
        messages=await generate.ainvoke(
        {"question": question, "context":documents, "user_data":user_data},config={"configurable": {"session_id": sessionid}}
        )
            # messages.append(message)

        return {"generation": [AIMessage(content=" ".join(messages))]}
    elif web_search=="yes" and nutritionRequired["nutrition"]=='yes':
        messages=await generator.ainvoke(
        {"question": question, "context":result, "user_data":user_data},config={"configurable": {"session_id": sessionid}}
        )
            # messages.append(message)
    else:
        print("not required")
        return {"generation": [AIMessage(content="")]}




async def grade_documents(state):
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
    nutritionRequired=state["nutritionBranch"]

    # Score each doc
    filtered_docs = []
    web_search = "no"
    print(nutritionRequired)
    if web_search=="yes" and nutritionRequired["nutrition"]=='yes':
        print("empeznado")
        for d in documents:
            print(d)
            score = await retrieval_grader.ainvoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            if grade== "no":
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "yes"
                break

        return {"documents": filtered_docs, "web_search": web_search}
    else:
        print("no required")
        return {"documents": [], "web_search": web_search}


# def transform_query(state):
#     """
#     Transform the query to produce a better question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Updates question key with a re-phrased question
#     """

#     print("---TRANSFORM QUERY---")
#     print(state)
#     question = state["question"]
#     documents = state["documents"]
#     sessionid= state["sessionid"]
#     user_data = state["userdata"]

#     # Re-write question
#     better_question = question_rewriter.invoke({"question": question})
#     return {"documents": documents, "question": better_question, "sessionid": sessionid, "userdata": user_data}


# def web_search(state):
#     """
#     Web search based on the re-phrased question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Updates documents key with appended web results
#     """
#     print("---WEB SEARCH---")
#     print(state)
#     question = state["question"]
#     documents = state["documents"]
#     sessionid=state["sessionid"]
#     user_data = state["userdata"]
#     # Web search
#     from langchain_community.retrievers import TavilySearchAPIRetriever

#     web_search_tool = TavilySearchAPIRetriever(k=3)
#     docs = web_search_tool.invoke(question)
#     documents.append(docs)
#     print("---WEB SEARCH RESULTS---")
#     print(docs)

#     return {"documents": documents, "question": question, "sessionid": sessionid, "user_data": user_data}


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
    print(state)
    web_search = state["web_search"]

    print("podria ser aqui")
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
    

#planner
async def get_plan(state):
    nutritionRequired= state["nutritionBranch"]
    web_search=state["web_search"]
    print(nutritionRequired.items())
    if web_search=="yes" and nutritionRequired["nutrition"]=="yes":
        print("empezando plan")
        task = state["question"]
        result = await planner.ainvoke({"task": task})
        print(result)
        # Find all matches in the sample text
        matches = re.findall(regex_pattern, result.content)
        print("matches: ", matches)
        return {"steps": matches, "plan_string": result.content}
    else:
        print("no required")
        return {"steps": [], "plan_string": ""}
    

#current task
def _get_current_task(state):
    if state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1
    
#Router
def route(state):
    _step = _get_current_task(state)
    if _step is None:
        # We have executed all tasks
        return "solve"
    else:
        # We are still executing tasks, loop back to the "tool" node
        return "tool"

# execute tools
async def tool_execution(state):
    nutritionRequired=state['nutritionBranch']
    web_search=state["web_search"]
    print(f"tool executor{nutritionRequired}")
    """Worker node that executes the tools of a given plan."""
    if web_search=="yes" and nutritionRequired["nutrition"]=='yes':
        print("empezando tool")
        _step = _get_current_task(state)
        print(_step)
        print(state["steps"])
        _, step_name, tool, tool_input = state["steps"][_step-1]
        _results = state["results"] or {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
        if tool == "Google":
            print("Google")
            result = await web_search_tool.ainvoke(tool_input)
        elif tool == "LLM":
            print("LLM")
            result = await llm.ainvoke(tool_input)
        else:
            raise ValueError
        _results[step_name] = str(result)
        return {"results": _results}
    else:
        print("no required")
        return {"results": ""}
    

async def solve(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---Solver---")
    question = state["question"]
    nutritionRequired=state["nutritionBranch"]
    web_search=state["web_search"]

    plan = ""
    print("puede ser por aca")
    if web_search=="yes" and nutritionRequired["nutrition"]=='yes' and state["steps"]!=[]:
        print("-------------Resolver-------------------")
        for _plan, step_name, tool, tool_input in state["steps"]:
            _results = state["results"] or {}
            for k, v in _results.items():
                tool_input = tool_input.replace(k, v)
                step_name = step_name.replace(k, v)
            plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
        prompt = solve_prompt.format(plan=plan, task=question)

        result = await llm.ainvoke(prompt)
        

        return {"result": result.content}
    else:
        return {"plan_string": plan}


async def recent_messages_add(state):
    """
    Generate common answers to the question

    Args:
    state (dict): The current graph state

    Returns:
    chat_vectorstore updated with recent messages
    """
    question = state["question"]
    sessionid= state["sessionid"]

    chat_message_history = MongoDBChatMessageHistory(
    session_id=sessionid,
    connection_string=mongo_uri,
    database_name="snapeatdb",
    collection_name="chatHistories",
    )

    decision=await memo_decider.ainvoke(input={"patient_message":question})
    print(decision)
    if decision["valuableinfo"]=='yes':

        add_lil_memo(question,memocreator_runnable=memo_creator)
    
    chat=chat_message_history.messages
    if len(chat)%5==0:
        print("creando resumen")
        summary(chat[-1:-5], summary_runnable=summary_chain)
        chat.clear()
        print("terminado resumen")

    return {"memoCreated":decision["valuableinfo"]}

