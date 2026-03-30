# Customer Support (Travel) Multi-Agent System Example

# - Functions: check_order_status, look_product_info
# - Agents: order_agent, product_agent, search_agent, root_agent
# - Tools on root: AgentTool(search_agent), check_order_status


import os
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool, google_search

load_dotenv()

groq_model = LiteLlm(
    model="groq/moonshotai/kimi-k2-instruct-0905", api_key=os.getenv("GROQ_API_KEY")
)


def check_order_status(order_id: str) -> dict:
    """
    Return the status of the travel booking (PNR/Reference)

    Input:
        order_id: str - booking reference or PNR number provided by the user. eg. "ABC123"

    Output (dict):
        {
            "order_id": order_id,
            "status": "Not Found" or "Confirmed" or "Cancelled",
            "pnr": "PNR123456" (if found),
            "from": "New York" (if found),
            "to": "Paris" (if found),
            "departure": "2024-07-01" (if found),
        }
    """
    print(f"Checking order status for order_id: {order_id}")
    booking_db = {
        "ABC123": {
            "status": "Confirmed",
            "pnr": "PNR123456",
            "from": "New York",
            "to": "Paris",
            "departure": "2024-07-01",
        },
        "DEF456": {
            "status": "Cancelled",
            "pnr": "PNR654321",
            "from": "Los Angeles",
            "to": "Tokyo",
            "departure": "2024-08-15",
        },
        "GHI789": {
            "status": "Not Found",
        },
        "JKL012": {
            "status": "Ticketed",
            "pnr": "PNR789012",
            "from": "Chicago",
            "to": "London",
            "departure": "2024-09-10",
        },
    }
    if order_id in booking_db:
        return {"status": "ok", "result": booking_db[order_id]}
    return {"status": "error", "message": "Order not found"}


def look_product_info(product_name: str) -> dict:
    """
    Return the destination/package details such as price, avaialbility, seasons, and highlights.
    Input:
        product_name: str - name of the travel package or destination. eg. "Paris"

    Output (dict):
        {
            "product_name": product_name,
            "avg_price": "$1000",
            "availability": "Available" or "Sold Out",
            "best_season": "Spring",
            "highlights": ["Eiffel Tower", "Louvre Museum", "Seine River Cruise"]
        }
    """
    print(f"Looking up product info for: {product_name}")
    product_db = {
        "paris": {
            "avg_price": "$1000",
            "availability": "Available",
            "best_season": "Spring",
            "highlights": ["Eiffel Tower", "Louvre Museum", "Seine River Cruise"],
        },
        "tokyo": {
            "avg_price": "$1200",
            "availability": "Available",
            "best_season": "Autumn",
            "highlights": ["Shibuya Crossing", "Senso-ji Temple", "Tokyo Skytree"],
        },
        "new york": {
            "avg_price": "$800",
            "availability": "Sold Out",
            "best_season": "Summer",
            "highlights": ["Statue of Liberty", "Central Park", "Times Square"],
        },
    }
    product_name = product_name.lower()
    if product_name in product_db:
        return {"status": "ok", "result": product_db[product_name]}
    return {"status": "error", "message": "Product not found"}


order_agent = LlmAgent(
    model=groq_model,
    name="order_agent",
    description="An agent that can check the status of a travel booking based on the provided order ID.",
    instruction="You are an order agent. Your task is to check the status of a travel booking based on the provided order ID. Use the check_order_status tool to fetch the booking information. Example: check_order_status('ABC123') Expected output: {'status': 'ok', 'result': {'order_id': 'ABC123', 'status': 'Confirmed', 'pnr': 'PNR123456', 'from': 'New York', 'to': 'Paris', 'departure': '2024-07-01'}}",
    tools=[check_order_status],
)

product_agent = LlmAgent(
    model=groq_model,
    name="product_agent",
    description="An agent that can provide information about travel destinations and packages based on the provided product name.",
    instruction="You are a product agent. Your task is to provide information about travel destinations and packages based on the provided product name. Use the look_product_info tool to fetch the destination/package details. Example: look_product_info('Paris') Expected output: {'status': 'ok', 'result': {'product_name': 'Paris', 'avg_price': '$1000', 'availability': 'Available', 'best_season': 'Spring', 'highlights': ['Eiffel Tower', 'Louvre Museum', 'Seine River Cruise']}}",
    tools=[look_product_info],
)

search_agent = LlmAgent(
    model=groq_model,
    name="search_agent",
    description="An Agent that can perform Google searches to answer user questions.",
    instruction="Use the google_search tool to perform Google searches and answer user questions in consize manner. Example: google_search('What are the top attractions in Paris?') Expected output: {'status': 'ok', 'result': 'The top attractions in Paris are the Eiffel Tower, Louvre Museum, and Seine River Cruise.'}",
    tools=[google_search],
)

root_agent = LlmAgent(
    model=groq_model,
    name="customer_service_agent",
    description="A customer service agent for a travel booking platform that can assist users with their travel-related queries, including checking booking status, providing information about travel destinations, and answering general questions using Google search.",
    instruction="You are a customer service agent for a travel booking platform. You can assist users with their travel-related queries, including checking booking status, providing information about travel destinations, and answering general questions using Google search. When a user query is received, determine the intent of the query and delegate it to the appropriate specialized agent (order_agent for booking status, product_agent for destination information, search_agent for general questions). If the query is about checking booking status, use order_agent. If the query is about travel destinations or packages, use product_agent. For all other queries, use search_agent.",
    sub_agents=[product_agent],
    tools=[AgentTool(agent=search_agent), check_order_status],
)
