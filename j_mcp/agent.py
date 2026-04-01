import json
import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import (
    McpToolset,
    StdioConnectionParams,
    StdioServerParameters,
)
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

load_dotenv()

groq_model = LiteLlm(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    # model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
    # api_key=os.getenv("OPENROUTER_API_KEY"),
)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable is not set")

# TAVILY_MCP_HEADERS = json.dumps({"Authorization": f"Bearer {TAVILY_API_KEY}"})

root_agent = LlmAgent(
    name="Tavily_MCP_Agent",
    model=groq_model,
    instruction="Use Tavily MCP Tools to find the relative web results",
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="tavily-mcp",
                    args=[],
                    env={"TAVILY_API_KEY": TAVILY_API_KEY},
                ),
                timeout=30.0,
            ),
        )
    ],
)
