FROM python:3.9.20-slim-bullseye

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn redis requests python-dotenv postgres psycopg2-binary pgvector \
    langchain-postgres psycopg[binary,pool] langchain langchain_openai langchain_core langchain_community

# wait for postgresql
RUN sleep 15

EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
