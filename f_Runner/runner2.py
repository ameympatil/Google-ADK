from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from agent import root_agent
from dotenv import load_dotenv
import asyncio
import uuid

load_dotenv()

APP_NAME = "QA Agent"
USER_ID = "user_1"
SESSION_ID = str(uuid.uuid4())

initial_state = {
    "name": "Alice",
    "information": {"age": 30, "hobbies": ["reading", "traveling", "cooking"]},
}

in_memory_session_service = InMemorySessionService()


async def main():
    new_session = await in_memory_session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state,
    )
    print(f"Session created with ID: {new_session}")

    runner = Runner(
        agent=root_agent,
        session_service=in_memory_session_service,
        app_name=APP_NAME,
    )

    input_message = types.Content(
        role="user",
        parts=[types.Part(text="What are the hobbies of the user?")],
    )

    for event in runner.run(
        user_id=USER_ID, session_id=SESSION_ID, new_message=input_message
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                print(f"Final response: {event.content.parts[0].text}")


asyncio.run(main())
