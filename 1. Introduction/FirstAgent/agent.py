from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import google_search


root_agent = LlmAgent(
    model='gemini-2.5-flash-lite',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
    tools=[google_search],
)
