from langchain_core.tools import tool
import openrouteservice
import requests
from shapely.geometry import Point, shape
from dotenv import load_dotenv
import os

load_dotenv()

ors_client = openrouteservice.Client(key=os.getenv("ORS_API_KEY"))
geoapify = os.getenv("GEOAPIFY_KEY")


def ors_isochrones(lat:float, lon:float, time_in_seconds: int = 1800)-> dict:
 """Calculates a walking polygon area within a given travel time limit.
    Args:
        lat: The latitude of the location as a float
        lon: The longitude of the location as a float
        time_in_seconds: The allowed walking time in integer seconds. Default is 1800
    Returns: GeoJSON dictionary representing the reachable walking zone."""
 response = ors_client.isochrones(
        locations=[[lon, lat]],
        profile='foot-walking',
        range=[time_in_seconds],
        range_type='time'
    )
 polygon_coordinates = response['features'][0]['geometry']['coordinates']
 return {
        "type": "Polygon",
        "coordinates": polygon_coordinates
    }


def geoapify_verify_location(place_name: str, city_context: str) -> dict:
    """Verifies a place name from Exa/SerpAPI and resolves it to coordinates.
    Also validates if the result actually falls within the intended city.
    Args:
        place_name: The name of the destination to be verified
        city_context: The city of travel where this location should exist
    """
    url = "https://api.geoapify.com/v1/geocode/search"
    query = f"{place_name}, {city_context}"
    params = {
        "text": query,
        "apiKey": geoapify,
        "limit": 1
    }
    res = requests.get(url, params=params).json()
    features = res.get("features", [])
    if not features:
        return {"found": False, "error": "Location could not be verified."}
    props = features[0]["properties"]
    return {
        "found": True,
        "name": props.get("name", place_name),
        "formatted_address": props.get("formatted"),
        "lat": props.get("lat"),
        "lon": props.get("lon"),
        "confidence": props.get("rank", {}).get("confidence"), # 0 to 1 confidence score
        "city": props.get("city")
    }


def shapely_destination_in_zone(dest_lat: float, dest_lon: float, zone_geojson: dict) -> bool:
    """Verifies if a destination point falls within Agent_Planner's walking boundary.
    Args:
        dest_lat: The latitude of the location as a float
        dest_lon: The longitude of the location as a float
        zone_geojson: The walkable zone provided by ors_isochromes in Agent_Planner
        """
    point = Point(dest_lon, dest_lat)
    polygon = shape(zone_geojson)
    return polygon.contains(point)


def ors_routing(origin: tuple, destination: tuple) -> dict:
    """Calculates public transport/walking route between two points.
    Strips raw geometry coordinates to optimize token size for LLM prompts.
    Args:
        origin: Origin location coordinates as tuple of floats formatted like (latitude, longitude)
        destination: Destination location coordinates as tuple of floats formatted like (latitude, longitude)
    """
    url = "https://api.geoapify.com/v1/routing"
    waypoints = f"{origin[0]},{origin[1]}|{destination[0]},{destination[1]}"
    params = {
        "waypoints": waypoints,
        "mode": "transit",  # Combines public transport + walking
        "apiKey": geoapify
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    res = response.json()
    if "features" in res and res["features"]:
        route = res["features"][0]["properties"]
    else:
        print("API Response did not contain 'features':", res)

    simplified_legs = []
    leg = route['legs']
    for step in leg[0]['steps']:
        instruction = step['instruction']['text']
        simplified_legs.append({
            "instruction": instruction
        })
    travel_mode = route['mode']
    duration = round((leg[0]['time']) / 60, 1)
    route_details = {'origin': origin, 'destination': destination, 'travel_mode': travel_mode, 'duration': duration, 'instructions': simplified_legs}   
    return {
        "routing": route_details
    }


def geoapify_map(destinations: list) -> str:
    """Generates an annotated map URL featuring numbered destination markers.
    Returns: Directly renderable PNG image URL.
    Args:
        destinations: A list item containing the coordinates of all locations mentioned in the final itinerary.
    """
    marker_strings = []
    for idx, item in enumerate(destinations, 1):
        # Format: lonlat:lng,lat;color:#hex;text:1
        marker_strings.append(f"lonlat:{item[1]},{item[0]};color:%23ff0055;text:{idx}")
    
    markers_param = "|".join(marker_strings)
    
    base_url = "https://maps.geoapify.com/v1/staticmap"
    map_url = f"{base_url}?style=osm-bright&width=600&height=400&marker={markers_param}&apiKey={geoapify}"
    
    return map_url

