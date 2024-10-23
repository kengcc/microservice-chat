# Restaurant Chatbot Project with Microservices Architecture

This a demo project for a GF BVM real time assistant that is distributed across several microservices. The virtual assistant can answer general questions about the Group Finance, Business Value Management Unit, such as MSAP, costing, princing and reporting.

## Services

The project consists of the following services:

1. **Frontend:** A React application that provides the user interface for interacting with the chatbot.

2. **chat-service:** A Python (FastAPI)-based backend that coordinates the communication between the frontend and chat-service. This service also manages the interaction history between the user and the chatbot.

3. **llm-service:** Another Python (FastAPI)-based backend hosting the chatbot algorithm. This service communicates with the AI engine (OpenAI GPT-3.5-turbo) to process user queries and generate suitable responses.

4. **Redis:** A Redis server used for state storage across the services.

5. **Postgres:** A Postgres server acting as a database for storing vector embeddings.


## Setup Env Variable
set your .env with OpenAI API Key and LangChain API Key.

## Run Application

To get the project up and running, make sure Docker is installed on your system.

Then, run the following command:

```bash
docker compose up --build
```

This command starts all services using the `compose.yml` file. It downloads the necessary Docker images, creates associated containers, and gets them running together.

## Data Population

The provided `insert_data.py` script can be used to populate the Postgres database with your data. To do this, run the script once the services are up and running. It will connect to the Postgres service, create the necessary tables, and insert data into them.

Before running `insert_data.py` script, you need to set the virtual enviroment and install the dependencies.
```
% python -m venv venv
% venv/script/active
% pip install -r requirement.txt
```

## Run Vitual Assistant

The virtual Assistant portal is running at
```
http://localhost:3000
```

