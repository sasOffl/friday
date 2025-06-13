import os

structure = {
    "backend": {
        "main.py": "",
        "config": {
            "__init__.py": "",
            "settings.py": "",
            "database.py": "",
        },
        "models": {
            "__init__.py": "",
            "database_models.py": "",
            "pydantic_models.py": "",
        },
        "crews": {
            "__init__.py": "",
            "data_ingestion_crew.py": "",
            "model_prediction_crew.py": "",
            "health_analytics_crew.py": "",
            "comparative_analysis_crew.py": "",
            "report_generation_crew.py": "",
        },
        "agents": {
            "__init__.py": "",
            "data_agents.py": "",
            "prediction_agents.py": "",
            "health_agents.py": "",
            "comparative_agents.py": "",
            "report_agents.py": "",
        },
        "tools": {
            "__init__.py": "",
            "market_data_tools.py": "",
            "sentiment_tools.py": "",
            "prediction_tools.py": "",
            "technical_tools.py": "",
            "visualization_tools.py": "",
        },
        "services": {
            "__init__.py": "",
            "analysis_service.py": "",
            "data_service.py": "",
            "websocket_service.py": "",
        },
        "utils": {
            "__init__.py": "",
            "logger.py": "",
            "exceptions.py": "",
            "helpers.py": "",
        },
        "routers": {
            "__init__.py": "",
            "analysis.py": "",
            "websocket.py": "",
        },
    },
    "frontend": {
        "index.html": "",
    },
    "data": {
        "stock_data.db": "",  # create empty file
        "chroma_db": {},      # directory only
    },
    "requirements.txt": "",
    "README.md": "",
}

def create_structure(base_path, struct):
    for name, content in struct.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)

if __name__ == "__main__":
    create_structure(".", structure)
    print("Directory structure created.")
