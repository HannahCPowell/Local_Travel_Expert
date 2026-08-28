from dotenv import load_dotenv

load_dotenv()

from tools.search_tool import google_local_search

res = google_local_search("Best breakfast cafe's in Tromsø, Norway", "Tromsø, Norway", "no", "no")
print("\n \n Google Local Search Tool Test \n Query: Best breakfast cafes Tromsø, Norway, Tromsø, Norway, no, no \n Results \n")
print(res)

from tools.search_tool import exa_semantic_search

res = exa_semantic_search("Best breakfast cafe's in Tromsø, Norway")
print("\n \n Exa Search Tool Test \n Query: Best breakfast cafes Tromsø, Norway \n Results \n")
print (res)

from tools.maps_tool import ors_isochrones

iso = ors_isochrones(lat=69.6480022, lon=18.9557948, time_in_seconds= 1800)
print("\n \n ORS Isochromes Tool Test \n Query: (lat=69.6480022, lon=18.9557948, time_in_seconds= 1800) \n Results \n")
print (iso)

from tools.maps_tool import geoapify_verify_location

res = geoapify_verify_location(place_name= "Comfort Hotel Xpress Tromsø", city_context= "Tromsø")
print("\n \n Geoapify Verify Location Tool Test \n Query: (place_name= Comfort Hotel Xpress Tromsø, city_context= Tromsø) \n Results \n")
print (res)

from tools.maps_tool import shapely_destination_in_zone

res = shapely_destination_in_zone(dest_lat= 69.6480022, dest_lon= 18.9557948, zone_geojson = iso)
print("\n \n Shapely Verify Destination in Zone Tool Test \n Query: Comparing coordinates of the cafe which was used to generate the zone against the generated zone (expected true) \n Results \n")
print (res)

res = shapely_destination_in_zone(dest_lat= 69.647576, dest_lon= 18.95236, zone_geojson = iso)
print("\n \n Shapely Verify Destination in Zone Tool Test \n Query: Comparing coordinates of the hotel against the zone generated around the cafe (expected true) \n Results \n")
print (res)

from tools.maps_tool import ors_routing

res = ors_routing(origin= (69.6480022, 18.9557948), destination= (69.647576, 18.95236))
print("\n \n ORS Routing Tool Test \n Query: Origin set to cafe coordinates, destination set to hotel coordinates \n Results \n")
print (res)

from tools.maps_tool import geoapify_map

res = geoapify_map(destinations= ((69.6480022, 18.9557948), (69.647576, 18.95236)))
print("\n \n Geoapify Map Creation Tool Test \n Query: Hotel and Cafe coordinates added to destinations list \n Results \n")
print (res)