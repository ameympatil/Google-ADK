from datetime import datetime
import logging
import os
import traceback
from typing import Any, Optional

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import BaseTool, ToolContext
from google.genai import types
from pyparsing import Dict

load_dotenv()

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_LOG_LEVEL = os.getenv("ADK_LOG_LEVEL", "INFO")


def configure_logger(
    name: str = __name__,
    level: str = DEFAULT_LOG_LEVEL,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create and configure a module-specific logger."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


logger = configure_logger()

groq_model = LiteLlm(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    # model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
    # api_key=os.getenv("OPENROUTER_API_KEY"),
)


# Agent Lifecycle
def before_agent_callback(
    callback_context: CallbackContext,
) -> Optional[CallbackContext]:
    try:
        logger.info("Before agent callback triggered")

        state = callback_context.state
        timestamp = datetime.now()

        if "agent_name" not in state:
            state["agent_name"] = "Simple Agent"
        if "request_counter" not in state:
            state["request_counter"] = 1
        else:
            state["request_counter"] += 1

        state["request_start_time"] = timestamp.isoformat()

        logger.debug(
            "agent_name=%s request_counter=%s",
            state["agent_name"],
            state["request_counter"],
        )
        logger.debug("request_start_time=%s", state["request_start_time"])

    except Exception as exc:
        logger.error(
            "Exception in before_agent_callback: %s\n%s",
            exc,
            traceback.format_exc(),
        )

    return None


def after_agent_callback(
    callback_context: CallbackContext,
) -> Optional[CallbackContext]:
    try:
        logger.info("After agent callback triggered")

        state = callback_context.state
        timestamp = datetime.now()

        if "request_start_time" not in state:
            logger.warning(
                "request_start_time missing in state for after_agent_callback"
            )
            return None

        state["request_end_time"] = timestamp.isoformat()
        state["duration"] = (
            datetime.fromisoformat(state["request_end_time"])
            - datetime.fromisoformat(state["request_start_time"])
        ).total_seconds()

        logger.debug(
            "request_end_time=%s duration=%s",
            state["request_end_time"],
            state["duration"],
        )

    except Exception as exc:
        logger.error(
            "Exception in after_agent_callback: %s\n%s",
            exc,
            traceback.format_exc(),
        )

    return None


def before_tool_callback(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    try:
        logger.info("Before tool callback: %s", tool.name)
        logger.debug("tool args: %s", args)

        tool_name = tool.name

        if (
            tool_name == "get_country_name"
            and args.get("country", "").lower() == "america"
        ):
            logger.info("Normalizing country name from America to United States")
            args["country"] = "United States"
            logger.debug("normalized args: %s", args)

        if (
            tool_name == "get_capital_city"
            and args.get("country", "").lower() == "restricted"
        ):
            logger.warning("Restricted country access attempted in get_capital_city")
            return {
                "result": "Access Denied: Restricted country information is not available."
            }

        logger.info("before_tool_callback completed for %s", tool.name)

    except Exception as exc:
        logger.error(
            "Exception in before_tool_callback for %s: %s\n%s",
            tool.name if tool else "unknown",
            exc,
            traceback.format_exc(),
        )
        # allow tools to still execute with unmodified arguments after error
        return None

    return None


def after_tool_callback(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Dict
) -> Optional[Dict]:
    try:
        logger.info("After tool callback: %s", tool.name)
        logger.debug("tool args: %s", args)
        logger.debug("tool response before modification: %s", tool_response)

        original_result = (tool_response or {}).get("result", "")

        if (
            tool.name == "get_capital_city"
            and isinstance(original_result, str)
            and original_result.lower() == "washington"
        ):
            modified_result = "Washington, D.C. (Modified by Callback)"
            logger.info(
                "Modifying get_capital_city result from '%s' to '%s'",
                original_result,
                modified_result,
            )
            tool_response["result"] = modified_result
            logger.debug("tool response after modification: %s", tool_response)

    except Exception as exc:
        logger.error(
            "Exception in after_tool_callback for %s: %s\n%s",
            tool.name if tool else "unknown",
            exc,
            traceback.format_exc(),
        )

    return None


root_agent = LlmAgent(
    name="root_agent",
    model=groq_model,
    description="A simple agent that uses Groq's Llama 3.3 70B Versatile model.",
    instruction="Answer the user's question to the best of your ability.",
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
)
