import json
from openai import OpenAI
import math

# Predefined list of cities along I-64 with their coordinates
i_64_route = [
    {"city": "Chesapeake, VA", "lat": 36.8190, "lon": -76.2749},
    {"city": "Richmond, VA", "lat": 37.5407, "lon": -77.4360},
    {"city": "Charlottesville, VA", "lat": 38.0293, "lon": -78.4767},
    {"city": "Lexington, VA", "lat": 37.7840, "lon": -79.4430},
    {"city": "Staunton, VA", "lat": 38.1496, "lon": -79.0717},
    {"city": "Charleston, WV", "lat": 38.3498, "lon": -81.6326},
    {"city": "Lexington, KY", "lat": 38.0406, "lon": -84.5037},
    {"city": "Louisville, KY", "lat": 38.2527, "lon": -85.7585},
    {"city": "Evansville, IN", "lat": 37.9748, "lon": -87.5558},
    {"city": "St. Louis, MO", "lat": 38.6270, "lon": -90.1994},
    {"city": "Wentzville, MO", "lat": 38.8139, "lon": -90.8462}
]

def get_coords(city_name):
    """Return (lat, lon) tuple for given city name in i_64_route; raises ValueError if not found."""
    for entry in i_64_route:
        if entry["city"].lower() == city_name.lower():
            return entry["lat"], entry["lon"]
    raise ValueError(f"City '{city_name}' not found in I-64 route.")

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance in miles between two latitude/longitude points."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# %%
# This is the JSON schema that tells the LLM what tools exist
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_coords",
            "description": "Get the latitude and longitude for a city on I-64",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "The full city name as on the I-64 route, e.g. 'Charlottesville, VA'"
                    }
                },
                "required": ["city_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_distance",
            "description": "Calculate the Haversine distance in miles between two latitude/longitude points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat1": {
                        "type": "number",
                        "description": "Latitude of the first location"
                    },
                    "lon1": {
                        "type": "number",
                        "description": "Longitude of the first location"
                    },
                    "lat2": {
                        "type": "number",
                        "description": "Latitude of the second location"
                    },
                    "lon2": {
                        "type": "number",
                        "description": "Longitude of the second location"
                    }
                },
                "required": ["lat1", "lon1", "lat2", "lon2"]
            }
        }
    }
]

# %%
def run_agent(user_query: str):
    """
    Agent that calculates the distance between two cities using haversine formula.
    The agent leverages tool use via function calling to:
    1. Get coordinates for each city.
    2. Calculate the distance between them.
    """
    client = OpenAI()
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. When asked to find the distance between two cities, "
            "use the provided tools to look up city coordinates and calculate Haversine distance. "
            "Never make up coordinates. Always use the functions."
        )},
        {"role": "user", "content": user_query}
    ]
    print(f"User: {user_query}\n")

    coords_for = {}
    for iteration in range(7):  # Allow more iterations: might need 2 tool calls before final answer
        print(f"--- Iteration {iteration + 1} ---")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        assistant_message = response.choices[0].message
        # If the model needs a tool, it returns assistant_message.tool_calls.
        # If it can answer directly, assistant_message.tool_calls is empty and it returns text.
        if assistant_message.tool_calls:
            print(f"LLM wants to call {len(assistant_message.tool_calls)} tool(s)")
            # Reformat the assistant message to classic dict format for OpenAI API
            messages.append({
                "role": assistant_message.role,
                "content": assistant_message.content,
                "tool_calls": [tc.model_dump() for tc in assistant_message.tool_calls]  # convert Pydantic models to dicts, if needed
            })
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")

                # Manual tool dispatch for city coordinate and distance resolution
                if function_name == "get_coords":
                    result = get_coords(**function_args)
                    # Save the result for use in distance calculation if needed
                    coords_for[function_args.get("city_name")] = result
                elif function_name == "calculate_distance":
                    result = calculate_distance(**function_args)
                else:
                    result = f"Error: Unknown function {function_name}"

                print(f"  Result: {result}")

                # Ensure result is JSON-serializable (tool message content must be a string)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result) if not isinstance(result, str) else result
                })

            print()
            # Loop again so LLM can combine tool results or finish response
        else:
            print(f"Assistant: {assistant_message.content}\n")
            return assistant_message.content

    return "Max iterations reached"

# %%
# ============================================
# PART 4: Test It
# ============================================

if __name__ == "__main__":
    # Build test_cases based on the provided I-64 route context (task3.ipynb lines 10-23)
    # We'll ask for coordinates for several cities on the route, a distance between two, and an out-of-route city for negative test
    test_cases = [
        ("I-64 city coordinates", "What are the coordinates of Chesapeake, VA?"),
        ("I-64 city coordinates", "What are the coordinates of Louisville, KY?"),
        ("I-64 city coordinates", "Give me the coordinates for St. Louis, MO."),
        ("Route spanning two states", "What is the distance from Charleston, WV to St. Louis, MO?"),
        # ("In-route and out-of-route", "Give me the coordinates for Evansville, IN and for Seattle, WA."),
        ("Just conversation", "List three cities along I-64 in the US."),
        ("Bad city", "What are the coordinates of Atlantis?"),
    ]

    N_REPEATS = 1  # Run each test this many times

    for i in range(N_REPEATS):
        print(f"\n{'='*60}")
        print(f" Test run {i+1} ".center(60, '='))
        print(f"{'='*60}\n")
        for idx, (desc, prompt) in enumerate(test_cases, 1):
            print("\n" + "="*60)
            print(f"TEST {idx}: {desc}")
            print("="*60)
            run_agent(prompt)


