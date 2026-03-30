import os
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.tools import AgentTool, google_search
from pydantic import BaseModel, Field

load_dotenv()

groq_model = LiteLlm(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    # model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
    # api_key=os.getenv("OPENROUTER_API_KEY"),
)


class TranslationOutput(BaseModel):
    spanish_translation: str = Field(..., description="Translated text in Spanish")
    french_translation: str = Field(..., description="Translated text in French")
    german_translation: str = Field(..., description="Translated text in German")


def get_agent(lang_code: str, output_key) -> LlmAgent:
    return LlmAgent(
        model=groq_model,
        name=f"translator_{lang_code}",
        description=f"A translator that can translate text to {lang_code}.",
        instruction=f"You are a translator that can translate text to {lang_code}. Translate the user input text to {lang_code}. Return only the translated text without any additional information. Example: translate('Hello, how are you?') Expected output: 'Hola, ¿cómo estás?'",
        output_key=output_key,
    )


# Agents
spanish_agent = get_agent("Spanish", "spanish_translation")
french_agent = get_agent("French", "french_translation")
german_agent = get_agent("German", "german_translation")

# root agent
translator = ParallelAgent(
    name="translator",
    description="A translator that can translate text to multiple languages.",
    sub_agents=[spanish_agent, french_agent, german_agent],
)

merger_agent = LlmAgent(
    model=groq_model,
    name="merger_agent",
    description="An agent that can merge translations from different languages into a single response.",
    instruction="You are a merger agent that can merge translations from different languages into a single response. The input will be a dictionary with language codes as keys and translated text as values. Merge the translations into a single json response that includes all translations. Example input: {'spanish_translation': 'Hola, ¿cómo estás?', 'french_translation': 'Bonjour, comment ça va?', 'german_translation': 'Hallo, wie geht es dir?'} Expected output: {'merged_translation': 'Spanish: Hola, ¿cómo estás? | French: Bonjour, comment ça va? | German: Hallo, wie geht es dir?'}",
    output_schema=TranslationOutput,
)

root_agent = SequentialAgent(
    name="root_agent",
    description="A root agent that can translate text to multiple languages and merge the translations into a single response.",
    sub_agents=[translator, merger_agent],
)
