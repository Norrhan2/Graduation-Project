"""
Database table definitions.

Three tables:
- Story: a source narrative (e.g. one Sira passage / event you're adapting)
- Scene: one comic panel's worth of content — the schema JSON from your
  fine-tuned LLM, plus pipeline status tracking
- nothing separate for images — the image URL lives directly on Scene,
  since it's a 1:1 relationship in this version of the pipeline
"""
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, Column, JSON


class Story(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    source_passage: str               # the raw Sira text this story came from
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Scene(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    story_id: int = Field(foreign_key="story.id")
    scene_number: int

    # The full schema JSON produced by your fine-tuned LLM (Layer 2 output).
    # Stored as JSON so you keep all fields (characters, setting, image_prompt, etc.)
    # without needing a column per field.
    schema_json: dict = Field(default={}, sa_column=Column(JSON))

    # Pipeline status: draft -> safety_checked -> image_generated -> published -> rejected
    status: str = Field(default="draft")

    image_url: Optional[str] = None
    safety_notes: Optional[str] = None   # why something was flagged, if it was

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)