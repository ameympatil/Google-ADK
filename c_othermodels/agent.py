# from google.adk.agents.llm_agent import Agent
import os
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent

load_dotenv()

groq_model = LiteLlm(
    model="groq/llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY")
)


def get_weather(location: str) -> dict:
    """Fetches the weather for a given location. Return a dictionary"""
    return {
        "status": "ok",
        "result": {"location": location, "temperature": "25°C", "condition": "Sunny"},
    }


root_agent = LlmAgent(
    model=groq_model,
    name="root_agent",
    description="A weather assistant that can answer user questions about the weather.",
    instruction="You are a weather assistant. Answer user questions about the weather. Use the get_weather tool to fetch weather information for specific location. Example: get_weather('New York') Expected output: {'status': 'ok', 'result': {'location': 'New York', 'temperature': '25°C', 'condition': 'Sunny'}}",
    tools=[get_weather],
    output_key="result",
)
