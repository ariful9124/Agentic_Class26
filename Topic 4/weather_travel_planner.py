"""
Smart Travel Planner - OpenWeatherMap Project

Agent takes: destination city + travel dates

Fetches: weather forecast for that specific date window

Generates: packing list + activity recommendations based on conditions

"""

import asyncio
import os
import requests
from datetime import datetime, timedelta, timezone
from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages


# ============================================================================
# CONFIGURATION
# ============================================================================

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


# ============================================================================
# TOOL DEFINITION 
# ============================================================================

@tool
def get_weather_forecast(
    city: str,
    start_date: str,
    end_date: str,
    units: str = "metric"
) -> str:
    """
    Fetch weather forecast for a specific city and date range.
    
    This tool retrieves weather data from OpenWeatherMap and filters it to the
    requested travel dates, providing temperature, conditions, and activity-relevant
    details like rain probability and wind.
    
    Args:
        city: Destination city name (e.g., "Paris", "Tokyo", "New York")
        start_date: Trip start date in YYYY-MM-DD format (e.g., "2024-03-15")
        end_date: Trip end date in YYYY-MM-DD format (e.g., "2024-03-18")
        units: Temperature units - "metric" (Celsius) or "imperial" (Fahrenheit)
        
    Returns:
        Formatted weather forecast for the specified date range including:
        - Daily temperature (high/low/feels-like)
        - Weather conditions (clear, rain, clouds, etc.)
        - Humidity and wind speed
        - Precipitation probability
    """
    if not OPENWEATHER_API_KEY:
        return "ERROR: OpenWeatherMap API key not set. Set OPENWEATHER_API_KEY environment variable."
    
    # Validate and parse dates
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if end < start:
            return "ERROR: end_date must be after start_date"
        
        # Track if we need to show a warning about date range
        date_range_warning = None
        if (end - start).days > 5:
            date_range_warning = "NOTE: OpenWeatherMap free tier provides 5-day forecast. Showing available data for first 5 days."
            
    except ValueError as e:
        return f"ERROR: Invalid date format. Use YYYY-MM-DD. Details: {e}"
    
    # Validate units
    if units not in ["metric", "imperial"]:
        return "ERROR: units must be 'metric' (Celsius) or 'imperial' (Fahrenheit)"
    
    # Make API request
    base_url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": units
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
    except requests.exceptions.HTTPError as e:
        if "401" in str(e):
            return "ERROR: Invalid API key. Check your OpenWeatherMap API key."
        elif "404" in str(e):
            return f"ERROR: City '{city}' not found. Check spelling or try a major city nearby."
        else:
            return f"ERROR: HTTP error: {str(e)}"
    except requests.exceptions.Timeout:
        return "ERROR: Request timed out. Check your internet connection."
    except requests.exceptions.RequestException as e:
        return f"ERROR: Network error: {str(e)}"
    except Exception as e:
        return f"ERROR: Unexpected error: {str(e)}"
    
    # Extract city info
    city_name = data["city"]["name"]
    country = data["city"]["country"]
    temp_unit = "°C" if units == "metric" else "°F"
    speed_unit = "m/s" if units == "metric" else "mph"
    
    # Filter forecast to requested date range
    forecast_summary = f"Weather Forecast for {city_name}, {country}\n"
    forecast_summary += f"Travel Dates: {start_date} to {end_date}\n"
    
    # Add warning if date range exceeds API limits
    if date_range_warning:
        forecast_summary += f"⚠️  {date_range_warning}\n"
    
    forecast_summary += "="*60 + "\n\n"
    
    # Group forecasts by day within the date range
    daily_forecasts = {}
    
    for item in data["list"]:
        # Use UTC to avoid timezone shifting near midnight
        forecast_dt = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
        forecast_date = forecast_dt.date()
        
        # Only include forecasts within the travel date range
        if start.date() <= forecast_date <= end.date():
            date_str = forecast_date.strftime("%Y-%m-%d")
            
            if date_str not in daily_forecasts:
                daily_forecasts[date_str] = {
                    "temps": [],
                    "feels_like": [],
                    "conditions": [],
                    "humidity": [],
                    "wind": [],
                    "rain_prob": 0,
                    "description": item["weather"][0]["description"]
                }
            
            daily_forecasts[date_str]["temps"].append(item["main"]["temp"])
            daily_forecasts[date_str]["feels_like"].append(item["main"]["feels_like"])
            daily_forecasts[date_str]["humidity"].append(item["main"]["humidity"])
            daily_forecasts[date_str]["wind"].append(item["wind"]["speed"])
            
            # Track rain probability
            if "pop" in item:
                daily_forecasts[date_str]["rain_prob"] = max(
                    daily_forecasts[date_str]["rain_prob"],
                    item["pop"] * 100
                )
            
            # Collect unique conditions
            condition = item["weather"][0]["main"]
            if condition not in daily_forecasts[date_str]["conditions"]:
                daily_forecasts[date_str]["conditions"].append(condition)
    
    # Check if we have data for the requested dates
    if not daily_forecasts:
        return f"ERROR: No forecast data available for dates {start_date} to {end_date}. OpenWeatherMap provides 5-day forecasts."
    
    # Format daily summaries
    for date_str in sorted(daily_forecasts.keys()):
        day_data = daily_forecasts[date_str]
        
        temp_high = max(day_data["temps"])
        temp_low = min(day_data["temps"])
        avg_feels_like = sum(day_data["feels_like"]) / len(day_data["feels_like"])
        avg_humidity = sum(day_data["humidity"]) / len(day_data["humidity"])
        avg_wind = sum(day_data["wind"]) / len(day_data["wind"])
        
        forecast_summary += f"📅 {date_str}:\n"
        forecast_summary += f"  Temperature: {temp_low:.1f}{temp_unit} to {temp_high:.1f}{temp_unit}\n"
        forecast_summary += f"  Feels like: {avg_feels_like:.1f}{temp_unit}\n"
        forecast_summary += f"  Conditions: {day_data['description']}\n"
        forecast_summary += f"  Humidity: {avg_humidity:.0f}%\n"
        forecast_summary += f"  Wind: {avg_wind:.1f} {speed_unit}\n"
        forecast_summary += f"  Rain probability: {day_data['rain_prob']:.0f}%\n"
        forecast_summary += "\n"
    
    return forecast_summary


# ============================================================================
# TOOL-ONLY TEST MODE 
# ============================================================================

def test_tool_only():
    """
    Test the weather tool in isolation before integrating with the agent.
    """
    print("="*80)
    print("HOUR 1: Testing Weather Tool in Isolation")
    print("="*80)
    
    if not OPENWEATHER_API_KEY:
        print("\n❌ ERROR: OpenWeatherMap API key not found!")
        print("Set it with: export OPENWEATHER_API_KEY='your_key_here'")
        print("Get a free key at: https://openweathermap.org/api")
        return
    
    print("\n✅ API key found. Testing tool...\n")
    
    # Test 1: Valid request
    print("Test 1: Valid request (Paris, 3-day trip)")
    print("-" * 60)
    
    # Calculate dates (tomorrow to 3 days from now)
    tomorrow = datetime.now() + timedelta(days=1)
    end_date = tomorrow + timedelta(days=2)
    
    result = get_weather_forecast.invoke({
        "city": "Paris",
        "start_date": tomorrow.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "units": "metric"
    })
    print(result)
    
    # Test 2: Imperial units
    print("\n" + "="*80)
    print("Test 2: Imperial units (New York)")
    print("-" * 60)
    
    result = get_weather_forecast.invoke({
        "city": "New York",
        "start_date": tomorrow.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "units": "imperial"
    })
    print(result)
    
    # Test 3: Error handling - invalid city
    print("\n" + "="*80)
    print("Test 3: Error handling (invalid city)")
    print("-" * 60)
    
    result = get_weather_forecast.invoke({
        "city": "InvalidCityXYZ123",
        "start_date": tomorrow.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "units": "metric"
    })
    print(result)
    
    # Test 4: Error handling - invalid dates
    print("\n" + "="*80)
    print("Test 4: Error handling (end before start)")
    print("-" * 60)
    
    result = get_weather_forecast.invoke({
        "city": "London",
        "start_date": "2024-03-20",
        "end_date": "2024-03-15",
        "units": "metric"
    })
    print(result)
    
    # Test 5: Date range > 5 days (should show warning but still return data)
    print("\n" + "="*80)
    print("Test 5: Long trip (7 days - should warn but show available data)")
    print("-" * 60)
    
    end_date_long = tomorrow + timedelta(days=6)
    
    result = get_weather_forecast.invoke({
        "city": "Rome",
        "start_date": tomorrow.strftime("%Y-%m-%d"),
        "end_date": end_date_long.strftime("%Y-%m-%d"),
        "units": "metric"
    })
    print(result)
    
    print("\n" + "="*80)
    print("✅ Tool testing complete! Ready for agent integration.")
    print("="*80)


# ============================================================================
# STATE DEFINITION 
# ============================================================================

class ConversationState(TypedDict):
    """State schema for the travel planner conversation."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    verbose: bool
    command: str  # "exit", "verbose", "quiet", or None


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def input_node(state: ConversationState) -> ConversationState:
    """Get input from user and handle special commands."""
    if state.get("verbose", False):
        print("\n" + "="*80)
        print("NODE: input_node")
        print("="*80)
    
    user_input = input("\nYou: ").strip()
    
    # Handle commands
    if user_input.lower() in ["quit", "exit"]:
        if state.get("verbose", False):
            print("[DEBUG] Exit command received")
        return {"command": "exit"}
    
    if user_input.lower() == "verbose":
        print("[SYSTEM] Verbose mode enabled")
        return {"command": "verbose", "verbose": True}
    
    if user_input.lower() == "quiet":
        print("[SYSTEM] Verbose mode disabled")
        return {"command": "quiet", "verbose": False}
    
    # Add message to conversation
    if state.get("verbose", False):
        print(f"[DEBUG] User input: {user_input}")
    
    return {"command": None, "messages": [HumanMessage(content=user_input)]}


def call_react_agent(state: ConversationState) -> ConversationState:
    """Invoke the ReAct agent with conversation history."""
    if state.get("verbose", False):
        print("\n" + "="*80)
        print("NODE: call_react_agent")
        print("="*80)
        print(f"[DEBUG] Invoking agent with {len(state['messages'])} messages")
    
    global react_agent
    
    messages_before = len(state["messages"])
    result = react_agent.invoke({"messages": state["messages"]})
    
    if state.get("verbose", False):
        new_count = len(result["messages"]) - messages_before
        print(f"[DEBUG] Agent generated {new_count} new messages")
        
        for msg in result["messages"][messages_before:]:
            if isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"[DEBUG] Tool calls: {[tc['name'] for tc in msg.tool_calls]}")
    
    new_messages = result["messages"][messages_before:]
    return {"messages": new_messages}


def output_node(state: ConversationState) -> ConversationState:
    """Display the assistant's response."""
    if state.get("verbose", False):
        print("\n" + "="*80)
        print("NODE: output_node")
        print("="*80)
    
    # Find last AI message
    last_ai_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            last_ai_message = msg
            break
    
    if last_ai_message:
        print(f"\n🤖 Travel Planner:\n{last_ai_message.content}")
    else:
        print("\n[WARNING] No response found")
    
    return {}


def trim_history(state: ConversationState) -> ConversationState:
    """Trim conversation history to prevent unbounded growth."""
    MAX_MESSAGES = 100
    
    if state.get("verbose", False):
        print("\n" + "="*80)
        print("NODE: trim_history")
        print("="*80)
    
    messages = state["messages"]
    current_count = len(messages)
    
    if current_count > MAX_MESSAGES:
        trimmed = messages[-MAX_MESSAGES:]
        if state.get("verbose", False):
            print(f"[DEBUG] Trimmed from {current_count} to {len(trimmed)} messages")
        return {"messages": trimmed}
    else:
        if state.get("verbose", False):
            print(f"[DEBUG] History: {current_count} messages (no trim needed)")
        return {}


# ============================================================================
# ROUTING
# ============================================================================

def route_after_input(state: ConversationState) -> str:
    """Route based on command field."""
    command = state.get("command")
    
    if command == "exit":
        if state.get("verbose", False):
            print("[DEBUG] Routing to END")
        return "end"
    
    if command in ["verbose", "quiet"]:
        if state.get("verbose", False):
            print("[DEBUG] Routing back to input")
        return "input"
    
    if state.get("verbose", False):
        print("[DEBUG] Routing to agent")
    return "call_react_agent"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

react_agent = None

def create_conversation_graph():
    """Build the conversation graph with structured travel planning output."""
    global react_agent
    
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # System prompt enforcing structured output format
    system_message = """You are a professional travel planning assistant with access to real-time weather forecasts.

When a user asks about travel plans, you MUST:
1. Extract the destination city and travel dates from their request
   - If dates are vague (e.g., "next week"), calculate specific dates
   - If no dates given, ask for them
2. Use the get_weather_forecast tool with the city, start_date, end_date, and units
3. Generate a response in this EXACT structure:

📍 WEATHER SNAPSHOT
[Brief overview of weather during their trip]

🎒 PACKING LIST
Essential items based on temperatures and conditions:
- [Item 1 with reason]
- [Item 2 with reason]
- [Item 3 with reason]
[... 8-12 items total]

🎯 RECOMMENDED ACTIVITIES
Based on weather conditions:
- [Activity 1]: [Why it's good for this weather]
- [Activity 2]: [Why it's good for this weather]
- [Activity 3]: [Why it's good for this weather]
[... 5-8 activities]

Always use the tool even if you think you know the weather. Provide specific, actionable advice."""
    
    react_agent = create_react_agent(
        model=model,
        tools=[get_weather_forecast],
        prompt=system_message
    )
    
    print("[SYSTEM] Travel planner agent created")
    
    # Build conversation graph
    workflow = StateGraph(ConversationState)
    
    workflow.add_node("input", input_node)
    workflow.add_node("call_react_agent", call_react_agent)
    workflow.add_node("output", output_node)
    workflow.add_node("trim_history", trim_history)
    
    workflow.set_entry_point("input")
    
    workflow.add_conditional_edges(
        "input",
        route_after_input,
        {
            "call_react_agent": "call_react_agent",
            "input": "input",
            "end": END
        }
    )
    
    workflow.add_edge("call_react_agent", "output")
    workflow.add_edge("output", "trim_history")
    workflow.add_edge("trim_history", "input")
    
    return workflow.compile()


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_graphs(wrapper_app):
    """Generate graph visualizations."""
    global react_agent
    
    try:
        react_png = react_agent.get_graph().draw_mermaid_png()
        with open("travel_planner_react_agent.png", "wb") as f:
            f.write(react_png)
        print("[SYSTEM] Agent graph → 'travel_planner_react_agent.png'")
    except Exception as e:
        print(f"[WARNING] Could not generate agent viz: {e}")
    
    try:
        wrapper_png = wrapper_app.get_graph().draw_mermaid_png()
        with open("travel_planner_conversation.png", "wb") as f:
            f.write(wrapper_png)
        print("[SYSTEM] Conversation graph → 'travel_planner_conversation.png'")
    except Exception as e:
        print(f"[WARNING] Could not generate conversation viz: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution: Full agent with interactive conversation."""
    print("="*80)
    print("Smart Travel Planner - Hour 2: Agent Integration")
    print("="*80)
    
    if not OPENWEATHER_API_KEY:
        print("\n❌ ERROR: OpenWeatherMap API key not found!")
        print("Set it: export OPENWEATHER_API_KEY='your_key_here'")
        print("Get key: https://openweathermap.org/api")
        return
    
    print("\n✅ API key found")
    print("\nFeatures:")
    print("  • Takes destination + travel dates")
    print("  • Fetches forecast for exact date window")
    print("  • Generates structured output:")
    print("    - Weather snapshot")
    print("    - Packing list with reasons")
    print("    - Activity recommendations")
    print("    - Bad-weather backup plan")
    print("\nCommands: 'verbose', 'quiet', 'quit'/'exit'")
    print("\nExample: 'I'm going to Paris March 15-18, what should I pack?'")
    print("="*80)
    
    app = create_conversation_graph()
    visualize_graphs(app)
    
    initial_state = {
        "messages": [],
        "verbose": False,
        "command": None
    }
    
    print("\n[SYSTEM] Starting conversation...\n")
    
    try:
        await app.ainvoke(initial_state)
    except KeyboardInterrupt:
        print("\n\n[SYSTEM] Interrupted (Ctrl+C)")
    
    print("\n[SYSTEM] Safe travels! 🌍✈️\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check for --test-tool flag 
    if len(sys.argv) > 1 and sys.argv[1] == "--test-tool":
        test_tool_only()
    else:
        asyncio.run(main())