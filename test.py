from tools.search_tool import google_local_search

res = google_local_search("Best breakfast cafe's in Tromsø, Norway", "Tromsø, Norway", "no", "no")
print(res)

# from tools.search_tool import exa_semantic_search

# res = exa_semantic_search("Best breakfast cafe's in Tromsø, Norway")
# print (res)

# from tools.maps_tool import google_maps_search

# res = google_maps_search("Comfort Hotel Xpress Tromsø", "Risø mat og kaffebar")
# print(res)

# from tools.maps_tool import ors_search

# res = ors_search([69.6480022, 18.9557948, 69.647576, 18.95236])
# print(res)