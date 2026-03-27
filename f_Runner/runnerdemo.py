from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agent import root_agent
from dotenv import load_dotenv
import asyncio

load_dotenv()

session_service_in_memory = InMemorySessionService()

initial_state = {
    "name": "Alice",
    "personalization": "I am Alice, a AI Engineer who loves to build AI applications. I am interested in badminton and cricket and my favourite players are Lakshya Sen and Virat Kohli.",
}

APP_NAME = "Answer Agent"
USER_ID = "user_123"
SESSION_ID = "session_123"


async def main():
    current_sessions = await session_service_in_memory.create_session(
        session_id=SESSION_ID, user_id=USER_ID, app_name=APP_NAME, state=initial_state
    )
    print(f"Session created with ID: {SESSION_ID}")

    created_sessions = await session_service_in_memory.list_sessions(
        user_id=USER_ID, app_name=APP_NAME
    )
    print(f"Created sessions: {created_sessions}")

    runner = Runner(
        agent=root_agent, session_service=session_service_in_memory, app_name=APP_NAME
    )

    new_message = types.Content(
        role="user",
        parts=[types.Part(text="Which is the favourite player of the user?")],
    )

    for event in runner.run(
        user_id=USER_ID, session_id=SESSION_ID, new_message=new_message
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                print(f"Final response: {event.content.parts[0].text}")


asyncio.run(main())
