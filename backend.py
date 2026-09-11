import os
import warnings
from typing import TypedDict, Annotated, List, Dict, Any, Union, Optional
import operator
from pydantic import Field, BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from psycopg.rows import dict_row
import uuid
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage
)
from langchain.rate_limiters import InMemoryRateLimiter
from langgraph.cache.memory import InMemoryCache
from langchain_openrouter import ChatOpenRouter
from tools.search_tool import google_local_search, exa_semantic_search
from tools.maps_tool import ors_isochrones, ors_routing, shapely_destination_in_zone, geoapify_verify_location, geoapify_map, ors_geocode
from dotenv import load_dotenv

load_dotenv()


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


node_cache = InMemoryCache()

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

llm = ChatOpenRouter(
    model="openrouter/free",
    temperature=0.2,
    rate_limiter= rate_limiter
)

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    location: str
    language: str
    country: str
    hotel_address: str
    hotel_lat: float
    hotel_lon: float
    dates: str
    plan_results: Dict[str, Any]
    foodie_results: Dict[str, Any]
    events_results: Dict[str, Any]
    sights_results: Dict[str, Any]
    itinerary_results: Dict[str, Any]
    final_results:Dict[str, Any]
    llm_calls: Annotated[int, operator.add]



# # =========================
# # Parser Agent
# # =========================



class TravelIntent(BaseModel):
    city: str = Field(description="Name of the target city e.g. 'Paris'")
    language: str = Field(description="2-letter ISO language code for the the primary national language of the destination")
    country: str = Field(description="2-letter ISO country code for the target destination")
    dates: str = Field(description="The dates and times of arrival and departure as provided.")
    hotel_address: str = Field(description="The street address of the hotel.")

llm_extractor = llm.with_structured_output(TravelIntent)

def agent_parser(state: TravelState) -> Dict[str, Any]:
    user_prompt = state.get("user_query")

    extraction_prompt = f"""
    From the user input ({user_prompt}) infer the 2-letter ISO country code for the target destination along with the target city.
    From the country, infer the primary national language and determine the 2-letter ISO language code for this.
    Determine the dates and times of travel with as much detail as is provided.
    Finally, determine the hotel name given and return its exact street address.
    """
    
    extracted_data = llm_extractor.invoke(extraction_prompt) 

    hotel_coordinates = ors_geocode(address = extracted_data.hotel_address, country = extracted_data.country)

    warnings.warn(f"Agent Parser : \n Hotel coordinates are {hotel_coordinates}. \n Extracted data: \n {extracted_data}.")
    return {
        "location": extracted_data.city,
        "language": extracted_data.language,
        "country": extracted_data.country,
        "hotel_address": extracted_data.hotel_address,
        "hotel_lat": hotel_coordinates[0],
        "hotel_lon": hotel_coordinates[1],
        "dates": extracted_data.dates,
        "llm_calls": 1
    }



# # =========================
# # Zone Planner Agent
# # =========================



class LocationCoordinates(BaseModel):
    name: str = Field(description="Name of the zone, landmark, or business")
    address: str = Field(description="Address of the location")
    summary: str = Field(description="Summary of what online users say about the location.")

class ExtractedZones(BaseModel):
    zones: List[LocationCoordinates] = Field(description="List of top 3 zones with coordinates")

def agent_planner(state: TravelState) -> Dict[str, Any]:
    user_query = state.get("user_query")
    city = state.get("location")
    country = state.get("country")
    search_query = f"top neighborhoods and zones of interest for {city}, {country}"
    language = state.get("language")
    hotel_lat = state.get("hotel_lat")
    hotel_lon = state.get("hotel_lon")
    
    exa_data = exa_semantic_search(search_query)
    google_data = google_local_search(
        query=search_query,
        location=city,
        language=language,
        country=country
    )

    llm_zones = llm.with_structured_output(ExtractedZones)
    
    extraction_prompt = f"""
    Analyze the following search results about places in {city}, {country}.
    Identify the top 3 distinct zones/neighborhoods of interest. 
    Provide their names, estimated center addresses. Provide the estimated center address as a street address.
    Provide a summary of why this location is interesting and what online users say about it.

    Exa Results: {exa_data}
    Google Results: {google_data}
    """
    extracted = llm_zones.invoke(extraction_prompt)

    zone_isochrones = []
    for zone in extracted.zones[:3]:
        if hotel_lat and hotel_lon >0 :
            zone_coordinates = ors_geocode(address = zone.address, country = country, center_lat= hotel_lat, center_lon= hotel_lon)
        else:
            zone_coordinates = ors_geocode(address = zone.address, country = country)
        # Calculate 30-min walking isochrome (1800 seconds)
        geo_polygon = ors_isochrones(lat=zone_coordinates[0], lon=zone_coordinates[1], time_in_seconds=1800)
        
        zone_isochrones.append({
            "zone_name": zone.name,
            "address": zone.address,
            "coordinates": {"lat": zone_coordinates[0], "lon": zone_coordinates[1]},
            "isochrome": geo_polygon
        })

    warnings.warn(f"Agent Planner : \n \n Only first three results are kept from the following : {extracted}.")
    return {
        "plan_results": zone_isochrones,
        "messages": [
            AIMessage(content=f"Successfully extracted {len(zone_isochrones)} zones and generated walking isochrones.")
        ],
        "llm_calls": 1
    }



# # =========================
# # Recommender Agents
# # =========================



class ExtractedSpots(BaseModel):
    spots: List[LocationCoordinates] = Field(description="List of top locations")

llm_spots = llm.with_structured_output(ExtractedSpots)

def agent_foodie(state: TravelState)-> Dict[str, Any]:
    city = state.get("location")
    language = state.get("language")
    country = state.get("country")
    plan_results = state.get("plan_results")
    zone_names = [z["zone_name"] for z in plan_results if "zone_name" in z]
    foodie_places = []
    for zone_name in zone_names:
        search_query = f"Top 3 bakeries and top 3 restaurants for {zone_name}, {city}, {country}. Give priority to local specialties and restaurants the locals love to hang out in."
        exa_data = exa_semantic_search(query=search_query)
        serpa_data = google_local_search(query=search_query, location=city, language=language, country=country)
        extraction_prompt = f"""
        Analyze the following search results about eateries in '{city}','{country}'.
        Identify the distinct restaurants and bakeries of interest. 
        Provide their names and exact street addresses.
        Provide a summary of why this location is interesting and what online users say about it.

        Exa Results: {exa_data}
        Google Results: {serpa_data}
        """
        extracted = llm_spots.invoke(extraction_prompt)

        if extracted is None:
            warnings.warn(f"LLM at Agent Foodie failed to return structured output matching the ExtractedSpots schema. No eateries gathered for zone {zone_name}")
            continue
        else:
            seen_names = set()
            unique_spots = []

            for spot in extracted.spots:
                    normalized_name = spot.name.strip().lower()
        
                    if normalized_name not in seen_names:
                        seen_names.add(normalized_name)
                        unique_spots.append(spot)

            extracted.spots = unique_spots

            warnings.warn(f"Agent Foodie: \n First extraction results \n \n {extracted}")

            for spot in extracted.spots[:10]:
                verify_location = geoapify_verify_location(spot.name, city)
                if verify_location is False:
                    warnings.warn(f"{spot.name} not found in {city}; skipped this eaterie!")
                    continue
                lat = verify_location.get("lat")
                lon = verify_location.get("lon")
                for zone in plan_results:
                    if lat and lon >1 :
                        verify_zone = shapely_destination_in_zone(lat, lon, zone_geojson= zone["isochrome"])
                        if verify_zone is True:
                            foodie_places.append({"name": spot.name, "address": verify_location["formatted_address"],"lat": lat, "lon": lon, "zone": zone["zone_name"], "summary": spot.summary})
                            break

    warnings.warn(f"Agent Foodie : \n \n Only first ten results are kept from the following : {unique_spots}.")
    return {        
        "foodie_results": foodie_places,
        "messages": [
            AIMessage(content=f"Successfully extracted {len(foodie_places)} foodie recommended spots.")
        ],
        "llm_calls": 1}


def agent_events(state: TravelState)-> Dict[str, Any]:
    city = state.get("location")
    language = state.get("language")
    country = state.get("country")
    dates = state.get("dates")
    plan_results = state.get("plan_results")
    events_places = []
    search_query = f"Top 5 events for {city} during this time interval: {dates}. Give priority to local cultural events, festivals, local live music, and free events."
    exa_data = exa_semantic_search(query=search_query)
    serpa_data = google_local_search(query=search_query, location=city, language=language, country=country)
    extraction_prompt = f"""
    Analyze the following search results about '{search_query}' in '{city}' during time window '{dates}'.
    Identify the distinct events of interest. 
    Provide their names and exact street addresses.
    Provide a summary of why this event is interesting and any other important information.

    Exa Results: {exa_data}
    Google Results: {serpa_data}
    """
    extracted = llm_spots.invoke(extraction_prompt)

    if extracted is None:
        warnings.warn(f"LLM at Agent Events failed to return structured output matching the ExtractedSpots schema. No events gathered.")
    else:
        seen_names = set()
        unique_spots = []

        for spot in extracted.spots:
                normalized_name = spot.name.strip().lower()

                if normalized_name not in seen_names:
                    seen_names.add(normalized_name)
                    unique_spots.append(spot)

        extracted.spots = unique_spots

        warnings.warn(f"Agent Events: \n First extraction results \n \n {extracted}")

        for spot in extracted.spots[:5]:
            verify_location = geoapify_verify_location(spot.name, city)
            if verify_location is False:
                warnings.warn(f"{spot.name} not found in {city}; skipped this event!")
                continue
            lat = verify_location.get("lat")
            lon = verify_location.get("lon")
            for zone in plan_results:
                if lat and lon >1 :
                    verify_zone = shapely_destination_in_zone(lat, lon, zone_geojson= zone["isochrome"])
                    if verify_zone is True:
                        events_places.append({"name": spot.name, "address": verify_location["formatted_address"],"lat": lat, "lon": lon, "zone": zone["zone_name"], "summary": spot.summary})
                    else:
                        events_places.append({"name": spot.name, "address": verify_location["formatted_address"],"lat": lat, "lon": lon, "zone": False, "summary": spot.summary})

    warnings.warn(f"Agent Events : \n \n Only first five results are kept from the following : {unique_spots}.")
    return {        
        "events_results": events_places,
        "messages": [
            AIMessage(content=f"Successfully extracted {len(events_places)} events recommended spots.")
        ],
        "llm_calls": 1}

def agent_sights(state: TravelState)-> Dict[str, Any]:
    city = state.get("location")
    language = state.get("language")
    country = state.get("country")
    plan_results = state.get("plan_results")
    zone_names = [z["zone_name"] for z in plan_results if "zone_name" in z]
    sights_places = []
    for zone_name in zone_names:
        search_query = f"Top 3 beautiful or cultural sights for {zone_name}, {city}, {country}. Include beautiful streets to walk down, or beautiful places to sit."
        exa_data = exa_semantic_search(query=search_query)
        serpa_data = google_local_search(query=search_query, location=city, language=language, country=country)
        extraction_prompt = f"""
        Analyze the following search results about sightseeing destinations in '{zone_name}', {city}, {country}.
        Identify the distinct sights of interest. 
        Provide their names and exact street addresses.
        Provide a summary of why this location is interesting and what online users say about it.

        Exa Results: {exa_data}
        Google Results: {serpa_data}
        """
        extracted = llm_spots.invoke(extraction_prompt)

        if extracted is None:
            warnings.warn(f"LLM at Agent Sights failed to return structured output matching the ExtractedSpots schema. No sights gathered in zone {zone_name}")
            continue
        else:
            seen_names = set()
            unique_spots = []

            for spot in extracted.spots:
                    normalized_name = spot.name.strip().lower()
        
                    if normalized_name not in seen_names:
                        seen_names.add(normalized_name)
                        unique_spots.append(spot)

            extracted.spots = unique_spots

            warnings.warn(f"Agent Sights: \n First extraction results \n \n {extracted}")

            for spot in extracted.spots[:5]:
                verify_location = geoapify_verify_location(spot.name, city)
                if verify_location is False:
                    warnings.warn(f"{spot.name} not found in {city}; skipped this sight!")
                    continue
                lat = verify_location.get("lat")
                lon = verify_location.get("lon")
                for zone in plan_results:
                    if lat and lon >1 :
                        verify_zone = shapely_destination_in_zone(lat, lon, zone_geojson= zone["isochrome"])
                        if verify_zone is True:
                            sights_places.append({"name": spot.name, "address": verify_location["formatted_address"],"lat": lat, "lon": lon, "zone": zone["zone_name"], "summary": spot.summary})
                            break

    warnings.warn(f"Agent Sights : \n \n Only first five results are kept from the following : {unique_spots}.")
    return {        
        "sights_results": sights_places,
        "messages": [
            AIMessage(content=f"Successfully extracted {len(sights_places)} sights recommended spots.")
        ],
        "llm_calls": 1}



# # =========================
# # Itinerary Agent
# # =========================



class SpotDetails(BaseModel):
    name: str = Field(description="Name of the zone, landmark, or business")
    opening_hours: str = Field(description="Hours of operation for the location.")
    avg_time_spent: str = Field(description="Average time spent at location.")
    best_time: str = Field(description="Best time of day to visit.")
    address: str = Field(description="Street address of the location")
    
class ItineraryItems(BaseModel):
    day: int = Field(description="Day of the trip")
    time_slot: str = Field(description="The approximate time frame (in military hours) when this item will be visited")
    name: str = Field(description="Name of the zone, landmark, or business")
    travel_time: int = Field(description="Transit time to reach the next point on the itinerary. If items are in the same zone, return zero.")
    address: str = Field(description="Street address of the location")

class Itinerary(BaseModel):
    items: List[ItineraryItems] = Field(description="List in chronological order of itinerary destinations.")

llm_items = llm.with_structured_output(SpotDetails)
llm_itinerary = llm.with_structured_output(Itinerary)

def agent_itinerary(state: TravelState) -> Dict[str, Any]:
    city = state.get("location")
    country = state.get("country")
    dates = state.get("dates")
    plan_results = state.get("plan_results", {})
    event_results = state.get("events_results")
    foodie_results = state.get("foodie_results")
    sights_results = state.get("sights_results")
    hotel_lat = state.get("hotel_lat")
    hotel_lon = state.get("hotel_lon")

    foodie_details = []
    for spot in foodie_results:
        extraction_prompt = f"""
        Analyze the following : {spot}
        Identify the opening hours, average time spent, and best time of day to visit.
        Include the street address provided in {spot}
        """
        extracted = llm_items.invoke(extraction_prompt)
        foodie_details.append({"foodie_results": foodie_results, "details": extracted})

    event_details = []
    for spot in event_results:
        extraction_prompt = f"""
        Analyze the following : {spot}
        Identify the opening hours, average time spent, and best time of day to visit.
        Include the street address provided in {spot}
        """
        extracted = llm_items.invoke(extraction_prompt)
        event_details.append({"event_results": event_results, "details": extracted})

    sight_details = []
    for spot in sights_results:
        extraction_prompt = f"""
        Analyze the following : {spot}
        Identify the opening hours, average time spent, and best time of day to visit.
        Include the street address provided in {spot}
        """
        extracted = llm_items.invoke(extraction_prompt)
        sight_details.append({"sight_results": sights_results, "details": extracted})
    
    warnings.warn(f"Agent Itinerary detailed examination found the following: \n\n {foodie_details} \n\n {event_details} \n\n {sight_details}")

    routing_destinations = []
    zones = [z["coordinates"] for z in plan_results]
    for zone in zones:
        lat = zone['lat']
        lon = zone['lon']
        routing_destinations.append({"coordinates": (lat,lon), "name": "zone"})

    routing_destinations.append({"coordinates": (hotel_lat, hotel_lon), "name": "hotel"})

    for spot in foodie_results:
        if spot['zone'] is False or spot['zone'] == "False":
            routing_destinations.append({"coordinates":(spot['lat'], spot['lon']), "name": spot['name']})
    
    for spot in event_results:
        if spot['zone'] is False or spot['zone'] == "False":
            routing_destinations.append({"coordinates":(spot['lat'], spot['lon']), "name": spot['name']})
    
    for spot in sights_results:
        if spot['zone'] is False or spot['zone'] == "False":
            routing_destinations.append({"coordinates":(spot['lat'], spot['lon']), "name": spot['name']})
    
    routes = []
    for origin in routing_destinations:
        others = [dest for dest in routing_destinations if dest != origin]
        for other in others:
            route = ors_routing(origin = origin['coordinates'], destination = other['coordinates'])
            routes.append({"route": route, "origin": origin['name'], "destination": other['name']})

    warnings.warn(f"Routes looks like this: \n\n {routes}")

    extraction_prompt_itinerary = f"""
    Determine an optimized itinerary for vacation in {city}, {country} during {dates}.
    Start and end each day in the hotel.
    Use the locations of interest in {foodie_details}, {event_details}, and {sight_details}. Group items by their zone to reduce travel time.
    Take note of transit times {routes} between zones and independent destinations.
    Include the street address provided for each location in the itinerary.
    """
    itinerary: Itinerary = llm_itinerary.invoke(extraction_prompt_itinerary)

    return {        
        "itinerary_results": itinerary.model_dump(),
        "messages": [
            AIMessage(content=f"Successfully generated itinerary with {len(itinerary.items)} spots.")
        ],
        "llm_calls": 1}



# # =========================
# # Final Travel Agent
# # =========================



def agent_assistant(state: TravelState) -> Dict[str, Any]:
    itinerary = state.get("itinerary_results", {})
    plans = state.get("plan_results")
    foodie = state.get("foodie_results")
    events = state.get("events_results")
    sights = state.get("sights_results")

    # if isinstance(itinerary, dict):
    #     items = itinerary.get("items", [])
    #     destinations = [item["address"] for item in items if "address" in item]
    # else:
    #     destinations = [item.address for item in itinerary.items]

    # map_url = geoapify_map(destinations= destinations)
    extraction_prompt = f"""
    You are a local travel expert, providing a beautiful travel itinerary.
    The itinerary is {itinerary}. More detailed thoughts on each location can be found in {plans}, {foodie}, {events}, and {sights}.
    """
    assistant_results = llm.invoke(extraction_prompt)

    content_text = getattr(assistant_results, "content", str(assistant_results))
    Final_Message = f"Local Travel Expert Message: \n\n {content_text}"
    # Final_Message = f"Find the annotated map here: \n {map_url} \n\n Local Travel Expert Message: \n\n {content_text}"
    return {
        "final_results": Final_Message,
        "messages": [AIMessage(content=f"Successfully generated final travel agent message. Bon voyage!")],
        "llm_calls": 1
    }



# # =========================
# # Graph
# # =========================



graph = StateGraph(TravelState)

graph.add_node("agent_parser", agent_parser)
graph.add_node("agent_planner", agent_planner)
graph.add_node("agent_foodie", agent_foodie)
graph.add_node("agent_events", agent_events)
graph.add_node("agent_sights", agent_sights)
graph.add_node("agent_itinerary", agent_itinerary)
graph.add_node("agent_assistant", agent_assistant)


graph.add_edge(START, "agent_parser")
graph.add_edge("agent_parser", "agent_planner")

graph.add_edge("agent_planner", "agent_foodie")
graph.add_edge("agent_planner", "agent_events")
graph.add_edge("agent_planner", "agent_sights")

graph.add_edge("agent_foodie", "agent_itinerary")
graph.add_edge("agent_events", "agent_itinerary")
graph.add_edge("agent_sights", "agent_itinerary")

graph.add_edge("agent_itinerary", "agent_assistant")
graph.add_edge("agent_assistant", END)



# # =========================
# # PostgreSQL Checkpointer
# # =========================



DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer, cache=node_cache)


# # =========================
# # Function for FastAPI
# # =========================



def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "location": "",
            "language": "",
            "country": "",
            "dates": "",
            "hotel_address": "",
            "hotel_lat": 0,
            "hotel_lon": 0,
            "plan_results": {},
            "foodie_results": {},
            "events_results": {},
            "sights_results": {},
            "itinerary_results": {},
            "final_results": {},
            "llm_calls": 0
        },
        config=config
    )

    return {
        "thread_id": thread_id,
        "llm_calls": result.get("llm_calls"),
        "location": result.get("location"),
        "language": result.get("language"),
        "country": result.get("country"),
        "dates": result.get("dates"),
        "hotel_address": result.get("hotel_address"),
        "hotel_lat": result.get("hotel_lat"),
        "hotel_lon": result.get("hotel_lon"),
        "plan_results": result.get("plan_results"),
        "foodie_results": result.get("foodie_results"),
        "events_results": result.get("events_results"),
        "sights_results": result.get("sights_results"),
        "itinerary_results": result.get("itinerary_results"),
        "final_results": result.get("final_results")
    }