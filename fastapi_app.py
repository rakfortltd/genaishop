from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag_engine import RAGEngine

app = FastAPI()
rag = RAGEngine()
rag.setup()

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    result = rag.ask(request.query)
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }
