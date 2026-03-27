# from google.adk.agents.llm_agent import Agent
import os
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent

load_dotenv()

groq_model = LiteLlm(
    model="groq/llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY")
)


root_agent = LlmAgent(
    model=groq_model,
    name="answer_agent",
    description="A helpful assistant for user queries.",
    instruction="You are a helpful assistant for user queries. Answer user questions to the best of your ability. You might have to answer questions about the user, their interests, or their preferences. You can access the same for user {name} in the session state. Personalization information is: \n {information}",
)
