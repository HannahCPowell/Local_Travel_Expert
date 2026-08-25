from dotenv import load_dotenv

load_dotenv()

from tools.search_tool import google_local_search

res = google_local_search("Best breakfast cafe's in Tromsø, Norway", "Tromsø, Norway", "no", "no")
print("Google Local Search Tool Test \n Query: Best breakfast cafes Tromsø, Norway, Tromsø, Norway, no, no \n Results \n")
print(res)

from tools.search_tool import exa_semantic_search

res = exa_semantic_search("Best breakfast cafe's in Tromsø, Norway")
print("Exa Search Tool Test \n Query: Best breakfast cafes Tromsø, Norway \n Results \n")
print (res)

from tools.maps_tool import google_maps_search

res = google_maps_search("Comfort Hotel Xpress Tromsø", "Risø mat og kaffebar")
print("Google Maps Tool Test \n Query: Comfort Hotel Xpress Tromsø, Risø mat og kaffebar \n Results \n")
print(res)

from tools.maps_tool import ors_search

res = ors_search([69.6480022, 18.9557948, 69.647576, 18.95236])
print("OpenRouteService Tool Test \n Query: 69.6480022, 18.9557948, 69.647576, 18.95236 \n Results \n")
print(res)