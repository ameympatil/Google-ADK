# from google.adk.agents.llm_agent import Agent
import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools import google_search, AgentTool

load_dotenv()


def get_weather(location: str) -> dict:
    """Fetches the weather for a given location. Return a dictionary"""
    return {
        "status": "ok",
        "result": {"location": location, "temperature": "25°C", "condition": "Sunny"},
    }


weather_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="weather_agent",
    description="A weather assistant that can answer user questions about the weather.",
    instruction="You are a weather assistant. Answer user questions about the weather. Use the get_weather tool to fetch weather information for specific location. Example: get_weather('New York') Expected output: {'status': 'ok', 'result': {'location': 'New York', 'temperature': '25°C', 'condition': 'Sunny'}}",
    tools=[get_weather],
)


search_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="search_agent",
    description="An Agent that can perform Google searches to answer user questions.",
    instruction="Use the google_search tool to perform Google searches and answer user questions in consize manner. Example: google_search('What is the capital of France?') Expected output: {'status': 'ok', 'result': 'The capital of France is Paris.'}",
    tools=[google_search],
)

tools = [AgentTool(agent=weather_agent), AgentTool(agent=search_agent)]


root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="A root agent that can delegate tasks to specialized agents.",
    instruction="You are a root agent. Delegate user questions to the appropriate specialized agent.",
    tools=tools,
)
