from google.adk.agents import LlmAgent, SequentialAgent
from pydantic import BaseModel, Field

class ResearchInfo(BaseModel):
    topic: str = Field(description="The topic of the research")
    summary: str = Field(description="A brief summary of the research findings")
    

topic_agent = LlmAgent(
    model='gemini-2.5-flash-lite',
    name='topic_agent',
    description='An agent to search for topic for potential research opportunities in AI',
    instruction='Search for potential research opportunities in AI and provide a topic',
    output_key='topic',
    output_schema=ResearchInfo,
)


summary_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='summary_agent',
    description='An agent to summarize research findings on a given topic',
    instruction='Summarize the research findings on the given topic',
    output_key='summary',
    output_schema=ResearchInfo,
)

root_agent = SequentialAgent(
    name='root_agent',
    description='A sequential agent to orchestrate the research process',
    sub_agents=[topic_agent, summary_agent],
)
