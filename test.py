from dotenv import load_dotenv

load_dotenv()

# from tools.search_tool import google_local_search

# res = google_local_search("Best breakfast cafe's in Tromsø, Norway", "Tromsø, Norway", "no", "no")
# print("\n \n Google Local Search Tool Test \n Query: Best breakfast cafes Tromsø, Norway, Tromsø, Norway, no, no \n Results \n")
# print(res)

# from tools.search_tool import exa_semantic_search

# res = exa_semantic_search("Best breakfast cafe's in Tromsø, Norway")
# print("\n \n Exa Search Tool Test \n Query: Best breakfast cafes Tromsø, Norway \n Results \n")
# print (res)

# from tools.maps_tool import ors_isochrones

# iso = ors_isochrones(lat=69.6480022, lon=18.9557948, time_in_seconds= 1800)
# print("\n \n ORS Isochromes Tool Test \n Query: (lat=69.6480022, lon=18.9557948, time_in_seconds= 1800) \n Results \n")
# print (iso)

# from tools.maps_tool import geoapify_verify_location

# res = geoapify_verify_location(place_name= "Comfort Hotel Xpress Tromsø", city_context= "Tromsø")
# print("\n \n Geoapify Verify Location Tool Test \n Query: (place_name= Comfort Hotel Xpress Tromsø, city_context= Tromsø) \n Results \n")
# print (res)

# from tools.maps_tool import shapely_destination_in_zone

# res = shapely_destination_in_zone(dest_lat= 69.6480022, dest_lon= 18.9557948, zone_geojson = iso)
# print("\n \n Shapely Verify Destination in Zone Tool Test \n Query: Comparing coordinates of the cafe which was used to generate the zone against the generated zone (expected true) \n Results \n")
# print (res)

# res = shapely_destination_in_zone(dest_lat= 69.647576, dest_lon= 18.95236, zone_geojson = iso)
# print("\n \n Shapely Verify Destination in Zone Tool Test \n Query: Comparing coordinates of the hotel against the zone generated around the cafe (expected true) \n Results \n")
# print (res)

# from tools.maps_tool import ors_routing

# res = ors_routing(origin= (69.6480022, 18.9557948), destination= (69.647576, 18.95236))
# print("\n \n ORS Routing Tool Test \n Query: Origin set to cafe coordinates, destination set to hotel coordinates \n Results \n")
# print (res)

# from tools.maps_tool import geoapify_map

# res = geoapify_map(destinations= ((69.6480022, 18.9557948), (69.647576, 18.95236)))
# print("\n \n Geoapify Map Creation Tool Test \n Query: Hotel and Cafe coordinates added to destinations list \n Results \n")
# print (res)





# ======================
# # Full Test
# ======================





import os
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
from langchain_openai import ChatOpenAI
from tools.search_tool import google_local_search, exa_semantic_search
from tools.maps_tool import ors_isochrones, ors_routing, shapely_destination_in_zone, geoapify_verify_location, geoapify_map
from dotenv import load_dotenv

load_dotenv()

GPT_API_KEY = os.getenv("GPT_API_KEY")
if not GPT_API_KEY:
    raise ValueError("GPT_API_KEY is missing. Please add it to your .env file.")

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    location: str
    language: str
    country: str
    hotel_address: str
    hotel_lat: float
    hotel_lon: float
    plan_results: Dict[str, Any]
    foodie_results: Dict[str, Any]
    events_results: Dict[str, Any]
    sights_results: Dict[str, Any]
    itinerary_results: Dict[str, Any]
    final_results:Dict[str, Any]
    llm_calls: Annotated[int, operator.add]

class TravelIntent(BaseModel):
    location: str = Field(description="Target city and country, e.g. 'Paris, France'")
    language: str = Field(description="2-letter ISO language code for the target destination")
    country: str = Field(description="2-letter ISO country code for the target destination")
    hotel_address: str = Field(description="The street address of the hotel.")
    hotel_lat: float = Field(description="Latitude coordinate of hotel as float.")
    hotel_lon: float = Field(description="Longitude coordinate of hotel as float.")

class LocationCoordinates(BaseModel):
    name: str = Field(description="Name of the zone, landmark, or business")
    address: str = Field(description="Address of the location")
    lat: float = Field(description="Latitude of the location")
    lon: float = Field(description="Longitude of the location")
    summary: str = Field(description="Summary of what online users say about the location.")

class ExtractedZones(BaseModel):
    zones: List[LocationCoordinates] = Field(description="List of top 3 zones with coordinates")

class ExtractedSpots(BaseModel):
    zones: List[LocationCoordinates] = Field(description="List of top locations with coordinates")

llm = ChatOpenAI(api_key= GPT_API_KEY, model="gpt-5.4-mini")
llm_extractor = llm.with_structured_output(TravelIntent)
llm_spots = llm.with_structured_output(ExtractedSpots)


# # # =========================
# # # Parser Agent
# # # =========================

# def agent_parser(state: TravelState) -> Dict[str, Any]:
#     user_prompt = state["user_query"]  # e.g. "I'm traveling to Paris next month"
    
#     # LLM extracts metadata from user prompt
#     extracted_data = llm_extractor.invoke(user_prompt) 
    
#     # Updates state so downstream agents have access
#     return {
#         "location": extracted_data.location,
#         "language": extracted_data.language,
#         "country": extracted_data.country,
#         "hotel_address": extracted_data.hotel_address,
#         "hotel_lat": extracted_data.hotel_lat,
#         "hotel_lon": extracted_data.hotel_lon,
#         "llm_calls": 1
#     }

# # =========================
# # Zone Planner Agent
# # =========================


location = "Tromsø, Norway"
language = "no"
country = "no"
# search_query = f"top neighborhood and zones of interest for {location}"
    
# exa_data = exa_semantic_search(search_query)
# google_data = google_local_search(
#     query=search_query,
#     location=location,
#     language=language,
#     country=country
# )

# llm_zones = llm.with_structured_output(ExtractedZones)

# extraction_prompt = f"""
# Analyze the following search results about interesting zones in '{location}'.
# Provide the names of the zones, estimated center addresses, and accurate latitude/longitude coordinates.
# Provide a summary of why this location is interesting and what online users say about it.

# Exa Results: {exa_data}
# Google Results: {google_data}
# """
# extracted: ExtractedZones = llm_zones.invoke(extraction_prompt)
# zone_isochrones = []
# for zone in extracted.zones:
#     # Calculate 30-min walking isochrome (1800 seconds)
#     geo_polygon = ors_isochrones(lat=zone.lat, lon=zone.lon, time_in_seconds=1800)
    
#     zone_isochrones.append({
#         "zone_name": zone.name,
#         "address": zone.address,
#         "coordinates": {"lat": zone.lat, "lon": zone.lon},
#         "isochrome": geo_polygon
#     })
# print(zone_isochrones)


locations = {"name": "Tromsøya / Central Tromsø", 'coordinates': {'lat': 69.6492, 'lon': 18.9553}, 'isochrome': {'type': 'Polygon', 'coordinates': [[[18.901318, 69.651993], [18.900347, 69.650889], [18.900292, 69.650789], [18.900712, 69.648491], [18.900588, 69.642643], [18.902817, 69.640009], [18.905598, 69.637822], [18.910021, 69.635842], [18.911983, 69.633312], [18.912982, 69.63145], [18.913111, 69.631426], [18.913646, 69.631447], [18.914083, 69.63151], [18.91732, 69.630616], [18.917433, 69.630578], [18.919355, 69.629526], [18.919586, 69.624403], [18.917111, 69.623197], [18.914636, 69.621992], [18.912161, 69.620786], [18.909732, 69.619602], [18.907279, 69.618491], [18.904825, 69.617379], [18.902372, 69.616268], [18.899919, 69.615157], [18.897465, 69.614045], [18.895012, 69.612934], [18.892559, 69.611823], [18.890105, 69.610711], [18.887652, 69.6096], [18.885199, 69.608488], [18.882745, 69.607377], [18.880288, 69.606264], [18.878238, 69.605329], [18.876189, 69.604394], [18.874146, 69.603462], [18.872103, 69.60253], [18.873597, 69.599254], [18.87564, 69.600187], [18.877683, 69.601119], [18.879733, 69.602054], [18.881782, 69.602988], [18.884231, 69.604098], [18.886684, 69.605209], [18.889137, 69.606321], [18.891591, 69.607432], [18.894044, 69.608543], [18.896497, 69.609655], [18.898951, 69.610766], [18.901404, 69.611877], [18.903857, 69.612989], [18.906311, 69.6141], [18.908764, 69.615212], [18.911217, 69.616323], [18.913738, 69.61755], [18.916213, 69.618755], [18.918687, 69.619961], [18.921162, 69.621166], [18.923637, 69.622372], [18.926112, 69.623577], [18.928614, 69.624872], [18.931077, 69.626147], [18.93354, 69.627422], [18.937178, 69.628503], [18.939592, 69.629753], [18.942007, 69.631003], [18.944474, 69.63228], [18.946942, 69.633558], [18.949409, 69.634835], [18.951877, 69.636113], [18.954344, 69.637391], [18.956812, 69.638668], [18.958444, 69.639583], [18.960021, 69.640469], [18.960936, 69.641138], [18.966058, 69.64423], [18.967692, 69.644312], [18.969928, 69.644759], [18.971617, 69.645474], [18.973416, 69.64624], [18.977702, 69.643013], [18.978431, 69.64138], [18.980821, 69.640999], [18.986057, 69.641085], [18.987128, 69.641135], [18.988064, 69.641428], [18.988511, 69.641573], [18.994109, 69.642369], [18.994829, 69.642251], [18.998226, 69.642637], [19.001581, 69.645397], [19.003114, 69.648654], [19.003435, 69.649506], [19.003357, 69.64964], [19.002867, 69.650199], [19.001829, 69.651054], [18.999648, 69.652356], [18.999304, 69.652441], [18.998364, 69.65233], [18.994719, 69.653872], [18.994716, 69.658506], [18.996736, 69.65999], [18.994604, 69.662891], [18.992585, 69.661407], [18.989981, 69.659814], [18.987424, 69.658341], [18.984828, 69.656845], [18.982232, 69.655349], [18.980076, 69.654117], [18.976013, 69.653576], [18.974301, 69.653907], [18.973794, 69.654074], [18.97036, 69.658549], [18.969433, 69.660727], [18.96999, 69.665174], [18.973538, 69.66557], [18.974958, 69.665482], [18.976803, 69.665923], [18.976935, 69.666099], [18.976902, 69.668296], [18.974872, 69.669702], [18.971945, 69.671164], [18.971405, 69.671342], [18.968784, 69.671672], [18.967929, 69.671694], [18.962944, 69.671692], [18.960055, 69.671203], [18.954642, 69.670833], [18.954434, 69.67082], [18.952289, 69.670514], [18.949311, 69.670049], [18.944607, 69.669609], [18.943105, 69.670292], [18.941581, 69.66991], [18.938059, 69.669051], [18.933869, 69.666681], [18.93022, 69.666419], [18.927404, 69.667026], [18.922468, 69.666129], [18.921494, 69.665978], [18.920477, 69.665601], [18.918022, 69.663722], [18.915913, 69.661996], [18.913647, 69.660415], [18.909483, 69.659451], [18.906302, 69.658555], [18.904396, 69.657043], [18.901318, 69.651993]]]}}

search_query = f"Top bakery for {locations['name']}. Give priority to local specialties and places the locals love to hang out in."
exa_data = exa_semantic_search(query=search_query)
serpa_data = google_local_search(query=search_query, location=location, language=language, country=country)
extraction_prompt = f"""
Analyze the following search results about '{search_query}' in '{locations['name']}'.
Identify the distinct bakeries of interest. 
Provide their names, street addresses, and accurate latitude/longitude coordinates.
Provide a summary of why this location is interesting and what online users say about it.

Exa Results: {exa_data}
Google Results: {serpa_data}
"""
extracted: ExtractedSpots = llm_spots.invoke(extraction_prompt)
# print(extracted.model_dump())

for spot in extracted.zones:
    verify_location = geoapify_verify_location(spot.name, location)
    if verify_location is False:
        continue
    verify_zone = shapely_destination_in_zone(verify_location["lat"], verify_location["lon"], zone_geojson= locations["isochrome"])
    if verify_zone is True:
        print({"name": spot.name, "address": verify_location["formatted_address"],"lat": verify_location["lat"], "lon": verify_location["lon"], "zone": locations["name"], "summary": spot.summary})
        break





