import os
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.tools import AgentTool, google_search
from pydantic import BaseModel, Field
from google.adk.tools.tool_context import ToolContext, CallbackContext

load_dotenv()

groq_model = LiteLlm(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    # model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
    # api_key=os.getenv("OPENROUTER_API_KEY"),
)


def exit_loop(tool_context: ToolContext):
    """
    A tool to exit the loop workflow. When called, it will signal the root agent to exit the loop and return an empty dictionary to the user.

    Input:
        tool_context: ToolContext - The context of the tool execution, which includes the current state of the agent and any relevant information.
    Output:
        A dictionary with a message indicating that the loop has been exited.
    """
    tool_context.actions.escalate = True
    return {}


initial_writer_agent = LlmAgent(
    model=groq_model,
    name="initial_writer_agent",
    description="An agent that writes blogs on the given topic. The output should be brief information about the given topic along with real-world examples and uses.",
    instruction="You are a blog writer that specializes in creating informative and engaging content on various topics. Your task is to write a brief overview of the given topic, including real-world examples and uses. Make some mistakes in grammar and keep the tone very simple alike a school kid writing the blog for the first time.",
    output_key="blog_content",
)

critic_agent = LlmAgent(
    model=groq_model,
    name="critic_agent",
    description="An agent that critiques the blog content written by the initial_writer_agent. The output should be a critique of the blog content, including suggestions for improvement.",
    instruction="You are a critic that evaluates the blog content written by the initial_writer_agent. Your task is to provide a critique of the blog content, including suggestions for improvement. Consider aspects such as clarity, engagement, informativeness, and relevance to the topic. If the blog content is as expected and no further improvements are needed, you may use the exit_loop tool at that point only to exit the loop. Below is the blog content to critique:\n {blog_content}",
    output_key="critique",
    tools=[exit_loop],
)

improver_agent = LlmAgent(
    model=groq_model,
    name="improver_agent",
    description="An agent that improves the blog content based on the critique provided by the critic_agent. The output should be the improved blog content.",
    instruction="You are an editor that enhances the blog content based on the critique provided by the critic_agent. Your task is to improve the blog content while maintaining its original intent and style. Consider the critique provided and make necessary adjustments to enhance clarity, engagement, informativeness, and relevance to the topic. Below is the original blog content and the critique:\n Original Blog Content: {blog_content}\n Critique: {critique}",
    output_key="blog_content",
)

loop_agent = LoopAgent(
    name="loop_agent",
    description="A loop agent that orchestrates the workflow of the blog writing process.",
    sub_agents=[critic_agent, improver_agent],
    max_iterations=5,
)

root_agent = SequentialAgent(
    name="root_agent",
    description="A root agent that manages the blog writing process by first generating content and then iteratively improving it based on critiques.",
    sub_agents=[initial_writer_agent, loop_agent],
)
