# Local_Travel_Expert
~A multi-agent travel planning tool which focuses on the local's perspective.~

## Description

This is a multi-agent AI tool built with LangChain which helps the user to plan a trip based on local search results. The recommendation agents use geographic and language markers from the destination to find unique ideas. These are ideas are considered along the map to create the most culturally-packed route possible. A supervisor agent manages the ideas and packages a final message to give you the best travel experience possible.

## Getting Started

### Dependencies

You will need to have API keys generated for the following services:
* LangSmith
* Eva
* SerpAPI
* Google Maps API
* OpenRouteService
* OpenRouter

A PostgreSQL server hosted by Render is connected to local pgAdmin, and the database URL is added to the .env file.

Here is a sample of the .env file contents:
```bash
ORS_API_KEY = "Your key"
GEOAPIFY_KEY = "Your key"
EXA_API_KEY = ""
SERPAPI_API_KEY = ""
DATABASE_URL = ""
LANGSMITH_PROJECT = ""
LANGSMITH_API_KEY = ""
LANGSMITH_TRACING = "true"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
OPENROUTER_API_KEY = ""
```

### Installing

1. Create Virtual Environment

```bash
conda create -n travel python=3.11 -y
```

2. Activate the environment

```bash
conda init zsh
conda activate travel
```

3. Install requirements

```bash
pip install -r requirements.txt
```

### Executing program

Initialize the API

```bash
python ./app.py
```
Follow the provided link to your local host server, and provide input to the agent.
Two example inputs are available in the 'Norway' and 'Spain' buttons.

## Authors

Hannah Powell
email: hannahcourtneypowell@proton.me

## License

This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE License - see the LICENSE.md file for details

## Acknowledgments

* [BAPPY AHMED - entbappy](https://github.com/entbappy/TripMate-AI-A-Multi-Agent-Travel-Planner-with-LangGraph)
