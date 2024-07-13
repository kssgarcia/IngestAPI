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
from .chains.memoryMethod.memory_management import summary, add_lil_memo


#mongo config

# Obtener la cadena de conexión desde una variable de entorno
MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")


# Obtener la cadena de conexión desde una variable de entorno
mongo_uri = f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.q4lvimh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

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

local_llm="llama3"



#Document loader
docuemt=document_loader(filename="TEMASDENUTRICINBSICABook.md")
#splitter
all_splits=text_splitter(docuemt)
#Embedding and vector store
vectorstore = store_and_retrieve(all_splits)

#tools
web_search_tool = TavilySearchAPIRetriever(k=3)
retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#Chains
generator=generator(local_llm=local_llm)
retrieval_grader=retrieval_grader(local_llm=local_llm)
question_rewriter=rewriter(local_llm=local_llm)
analyser=analyser(local_llm=local_llm)
branchDecider=branchDecider(local_llm=local_llm)
commonGenerator=commonGenerator(local_llm=local_llm,)
planner=plannerchain(local_llm=local_llm)

memo_decider=memo_decider(local_llm=local_llm)


def inicialize(state):
    question=state["question"]
    sessionid= state["sessionid"]
    userdata= state["userdata"]
    nutritionRequired=state["nutritionBranch"]
    documents=state["documents"]
    """     
    Initializes the state of the document
    """
    return {"question":question, "sessionid":sessionid, "userdata":userdata, "nutritionBranch":nutritionRequired, "documents":documents}

def branch(state):
    """
    Decides wether documents are needed or not based on the question

    Args:
    state (Document): The current state of the document

    Returns:
    state (Document): The new state with a new nuutritionalBranch key
    """
    print("---BRANCH---")
    question=state["question"]
    userdata= state["userdata"]
    diagnosis = state["diagnosis"]
    sessionid= state["sessionid"]

    required = branchDecider.invoke({"sentence":question})
    return {"question":question, "userdata":userdata, "nutritionBranch":required, "sessionid":sessionid, "diagnosis":diagnosis}

def nutritionRequired(state):
    """
    Decides whether the given question requires nutritional information

    Args:
        state (dict): The current graph state

    Returns:ß
        Binary decision, generate or go into the nutritional branch
    """
    print("---Nutrition Required---")
    question = state["question"]
    userdata= state["userdata"]
    required=state["nutritionBranch"]
    diagnosis = state["diagnosis"]
    sessionid= state["sessionid"]

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
    userdata= state["userdata"]
    required=state["nutritionBranch"]
    diagnosis = state["diagnosis"]
    sessionid= state["sessionid"]


    
    print("---GENERATE COMMON---")
    messages=[]
    async for message in commonGenerator.astream(input={"question":question}, config={"configurable": {"session_id": sessionid}}):
         messages.append(message)
    return {"question":question, "generation":[AIMessage(content=" ".join(messages))], "userdata":userdata, "nutritionBranch":required, "sessionid":sessionid, "diagnosis":diagnosis}




def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:ß
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    user_data=state["userdata"]
    diagnosis = state["diagnosis"]
    sessionid= state["sessionid"]

    # Retrieval
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question, "userdata": user_data, "sessionid": sessionid, "diagnosis": diagnosis}



async def formatuserdata(state):
    """ 
    Analyze user data and format it for it to be used in the model.

    Args:
     state (dict): The current graph state
    
    Returns:
        state (dict): New key added to state, userdata, that contains formatted user data
    """
    print("---FORMATUSERDATA---")
    user_data = state["userdata"]
    documents=state["documents"]
    question=state["question"]
    sessionid=state["sessionid"]

    # Format user data
    formatted_data = (
        f"name:{user_data.name},"
        f"lastName:{user_data.lastName},"
        f"Email: {user_data.email}, "
        f"Age: {user_data.dietetics.age}, "
        f"Lifestyle: {user_data.dietetics.lifestyle}, "
        f"Job: {user_data.dietetics.job}, "
        f"Anthropometry: (Height: {user_data.dietetics.anthropometry.height}, "
        f"Weight: {user_data.dietetics.anthropometry.weight}, "
        f"BMI: {user_data.dietetics.anthropometry.BMI}, "
        f"Waist Circumference: {user_data.dietetics.anthropometry.waist_circumference}), "
        f"Biochemical Indicators: (Glucose: {user_data.dietetics.biochemical_indicators.glucose}, "
        f"Cholesterol: {user_data.dietetics.biochemical_indicators.cholesterol}), "
        f"Diet: (Ingest Preferences: {', '.join(user_data.dietetics.diet.ingest_preferences)}, "
        f"Fruits and Vegetables: {user_data.dietetics.diet.fruits_and_vegetables}, "
        f"Fiber: {user_data.dietetics.diet.fiber}, "
        f"Saturated Fats: {user_data.dietetics.diet.saturated_fats}, "
        f"Sugars: {user_data.dietetics.diet.sugars}, "
        f"Today Meals: {', '.join(user_data.dietetics.diet.today_meals)}), "
        f"Social Indicators: (Marital Status: {user_data.dietetics.social_indicators.marital_status}, "
        f"Income: {user_data.dietetics.social_indicators.income}, "
        f"Access to Healthy Foods: {user_data.dietetics.social_indicators.access_to_healthy_foods}), "
        f"Physical Activity: {user_data.dietetics.physical_activity}, "
        f"Daily Activities: {', '.join(user_data.dietetics.daily_activities)}, "
        f"Activities Energy Consumption: {user_data.dietetics.activities_energy_consumption}, "
        f"Goals: (Reduce Weight: {user_data.dietetics.goals}, "
        f"Potential Diseases: (Type 2 Diabetes: {user_data.dietetics.potential_diseases}, "
    )

    messages=[]
    async for message in analyser.astream(input={formatted_data}):
        messages.append(message)
    return {"userdata": user_data, "question": question, "documents": documents,"diagnosis":'si',"generation":[AIMessage(content=" ".join(messages))], "sessionid": sessionid}



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
    user_data = state["userdata"]
    sessionid= state["sessionid"]

    # RAG generation
    messages=[]	
    # RAG generation
    async for message in generate.astream(
    {"question": question, "context":documents, "user_data":user_data},
    {"configurable": {"session_id": sessionid}},):
        messages.append(message)
    # generation = rag_chain.invoke({"context": documents, "question": question, "messages":chat_history.messages})
    # chat_history.add_ai_message(generation)
    # print(generation)
    return {"documents": documents, "question": question, "generation": [AIMessage(content=" ".join(messages))], "sessionid":sessionid}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    print(state)
    question = state["question"]
    documents = state["documents"]
    sessionid= state["sessionid"]
    user_data = state["userdata"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        print(d)
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search, "sessionid": sessionid,"userdata": user_data}


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
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]
    sessionid=state["sessionid"]

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
def get_plan(state):
    if state["web_search"]=="yes":
        task = state["question"]
        result = planner.invoke({"task": task})
        print(result)
        # Find all matches in the sample text
        matches = re.findall(regex_pattern, result.content)
        print("matches: ", matches)
        return {"steps": matches, "plan_string": result.content}
    else:
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
def _route(state):
    _step = _get_current_task(state)
    if _step is None:
        # We have executed all tasks
        return "solve"
    else:
        # We are still executing tasks, loop back to the "tool" node
        return "tool"

# execute tools
def tool_execution(state):
    web_search=state['web_search']
    nutritionRequired=state['nutritionBranch']
    """Worker node that executes the tools of a given plan."""
    if web_search=="yes" and nutritionRequired["nutrition"]=='yes':
        _step = _get_current_task(state)
        print(_step)
        print(state["steps"])
        _, step_name, tool, tool_input = state["steps"][_step-1]
        _results = state["results"] or {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
        if tool == "Google":
            print("Google")
            result = web_search.invoke(tool_input)
        elif tool == "LLM":
            print("LLM")
            result = llm.invoke(tool_input)
        else:
            raise ValueError
        _results[step_name] = str(result)
        return {"results": _results}
    else:
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
    documents = state["documents"]
    diagnosis= state["diagnosis"]
    web_search=state["web_search"]
    user_data = state["userdata"]
    sessionid= state["sessionid"]
    image=state["image_file"]
    nutritionRequired=state["nutritionBranch"]

    plan = ""
    
    if web_search=="yes" and nutritionRequired["nutrition"]=='yes' and state["steps"]!=[]:
        print("-------------Resolver-------------------")
        for _plan, step_name, tool, tool_input in state["steps"]:
            _results = state["results"] or {}
            for k, v in _results.items():
                tool_input = tool_input.replace(k, v)
                step_name = step_name.replace(k, v)
            plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
        prompt = solve_prompt.format(plan=plan, task=question)

        result = llm.invoke(prompt)
        messages=[]	
        async for message in generator.astream(
        {"question": question, "context":result, "user_data":diagnosis},
        {"configurable": {"session_id": sessionid}},):
            messages.append(message)

        return {"result": result.content,"generation": [AIMessage(content=" ".join(messages))]}
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
    userdata= state["userdata"]
    required=state["nutritionBranch"]
    diagnosis = state["diagnosis"]
    sessionid= state["sessionid"]
    image=state["image_file"]   

    chat_message_history = MongoDBChatMessageHistory(
    session_id=sessionid,
    connection_string=mongo_uri,
    database_name="snapeatdb",
    collection_name="chatHistories",
    )

    decision=memo_decider.invoke(input={"patient_message":question})
    if decision["valuableinfo"]=='yes':
        add_lil_memo(question)
    
    chat=chat_message_history.messages
    if len(chat)%5==0:
        summary(chat[-1:-5])
        chat.clear()

    return {"memoCreated":decision["valuableinfo"]}

