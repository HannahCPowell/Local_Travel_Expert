from langchain_core.tools import tool
import googlemaps
import openrouteservice
from dotenv import load_dotenv

load_dotenv()

gmaps_client = googlemaps.Client(key="GOOGLEMAPS_API_KEY")
ors_client = openrouteservice.Client(key="ORS_API_KEY")

def google_maps_search(origin:str, destination:str) -> dict:
 """Get public transport routes between destinations using Google Maps."""
 return gmaps_client.directions(origin, destination, mode='transit')

def ors_search(coordinates: list) -> dict:
 """Get routing geometry and walking routes using OpenRouteService."""
 return ors_client.directions(coordinates=coordinates, profile='foot-walking')

all_map_tools = [google_maps_search, ors_search]