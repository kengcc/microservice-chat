# Copyright (c) 2024 Telekom Research and Development Sdn BHd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.
#

from langchain_community.vectorstores import PGVector
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import find_dotenv, load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import logging


load_dotenv(find_dotenv())
## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
CONNECTION_STRING = "postgresql+psycopg://langchain:langchain@postgres:5432/langchain"
COLLECTION_NAME="vectordb"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

## Design ChatPrompt Template
prompt = ChatPromptTemplate.from_template("""
As a virtual assistant for Business Value Management (BVM), you have the following information related to BVM department. 
Think step by step before providing a detailed answer. 
<context>
{context}
</context>
Question: {input}""")

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

ROLE_CLASS_MAP = {
    "assistant": AIMessage,
    "user": HumanMessage,
    "system": SystemMessage
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    conversation: List[Message]

def create_messages(conversation):
    return [ROLE_CLASS_MAP[message.role](content=message.content) for message in conversation]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## Create Stuff Docment Chain
llm=ChatOpenAI(model="gpt-3.5-turbo")
document_chain=create_stuff_documents_chain(llm,prompt)

retriever=store.as_retriever()
retrieval_chain=create_retrieval_chain(retriever,document_chain)

@app.post("/llm_service/{conversation_id}")
async def llm_service(conversation_id: str, conversation: Conversation):
    query = conversation.conversation[-1].content
    response=retrieval_chain.invoke({"input": query})
    # print (response['answer'])

    return {"id": conversation_id, "reply": response['answer']}
