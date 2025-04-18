import chainlit as cl
from rag_engine import RAGEngine

@cl.on_chat_start
async def start():
    rag = RAGEngine()
    rag.setup()
    cl.user_session.set("rag", rag)
    await cl.Message("RAG system ready! Ask your question 📄🤖").send()

@cl.on_message
async def handle_message(message: cl.Message):
    rag: RAGEngine = cl.user_session.get("rag")
    result = rag.ask(message.content)

    sources = "\n".join(f"- {src}" for src in result["sources"])
    await cl.Message(
        content=f"**Answer:** {result['answer']}\n\n**Sources:**\n{sources}"
    ).send()
