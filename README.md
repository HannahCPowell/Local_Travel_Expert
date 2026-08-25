# Local_Travel_Expert
~A multi-agent travel planning tool which focuses on the local's perspective.~

## Description

This is a milti-agent AI tool built with LangSmith which helps the user to plan a trip based on local search results. The recommendation agents use geographic and language markers from the destination to find unique ideas. These are ideas are considered along the map to create the most culturally-packed route possible. A supervisor agent manages the ideas and packages a final message along with an annotated map to give you the best travel experience possible.

## Getting Started

### Dependencies

You will need to have API keys generated for the following services:
* LangSmith
* Eva
* SerpAPI
* Google Maps API
* OpenRouteService
* OpenAI

A PostgreSQL server hosted by Render is connected to local pgAdmin, and the database URL is added to the .env file.

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

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Hannah Powell
email: hannahcourtneypowell@proton.me

## License

This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE License - see the LICENSE.md file for details

## Acknowledgments

* [BAPPY AHMED - entbappy](https://github.com/entbappy/TripMate-AI-A-Multi-Agent-Travel-Planner-with-LangGraph)
