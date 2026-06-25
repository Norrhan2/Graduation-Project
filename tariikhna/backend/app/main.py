"""
Entrypoint for the Tariikhna backend.

Run from the backend/ folder with:
    uvicorn app.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""
from fastapi import FastAPI
from app.database import init_db
from app.routers import stories, scenes

app = FastAPI(
    title="Tariikhna API",
    description="Converts Islamic historical narratives into child-friendly comic panels.",
    version="0.1.0",
)

app.include_router(stories.router)
app.include_router(scenes.router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def health_check():
    return {"status": "ok", "service": "tariikhna-backend"}