from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from dotenv import load_dotenv

load_dotenv()

def create_agent(k=5):
    search = TavilySearchResults(k=k)
    tools = [search]
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # Hub에서 프롬프트를 가져옵니다
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # ChatPromptTemplate에 agent_scratchpad를 추가합니다
    messages = prompt.messages + [
        HumanMessagePromptTemplate.from_template("Additional Input: {input}"),
        AIMessagePromptTemplate.from_template("Scratchpad: {agent_scratchpad}")
    ]
    
    prompt_with_scratchpad = ChatPromptTemplate.from_messages(messages)

    agent = create_openai_functions_agent(llm, tools, prompt_with_scratchpad)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor