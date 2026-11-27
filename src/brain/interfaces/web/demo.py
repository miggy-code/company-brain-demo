"""The persona demo's display data (`businesses/<id>/demo.yaml`): persona blurbs, the personas
shown first, and suggested questions, each in Spanish and English. Who the personas are
(name, title, role) comes from the persons table, never from here."""

from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from brain.tenancy.config import ConfigError, Slug, read_yaml, validate

MAX_PERSONAS = 4


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Copy(_Strict):
    es: str = Field(min_length=1, max_length=300)
    en: str = Field(min_length=1, max_length=300)


class DemoPersona(_Strict):
    id: Slug
    blurb: Copy


class DemoConfig(_Strict):
    personas: Annotated[tuple[DemoPersona, ...], Field(min_length=1)]
    defaults: Annotated[tuple[Slug, ...], Field(min_length=1, max_length=MAX_PERSONAS)]
    questions: Annotated[tuple[Copy, ...], Field(min_length=1, max_length=8)]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        ids = [p.id for p in self.personas]
        if len(set(ids)) != len(ids):
            raise ValueError("a persona is listed twice")
        if not set(self.defaults) <= set(ids) or len(set(self.defaults)) != len(self.defaults):
            raise ValueError("defaults must be distinct listed personas")
        return self


def load_demo(path: Path, person_ids: set[str]) -> DemoConfig:
    """The demo config, checked against the business's people."""
    demo = validate(DemoConfig, read_yaml(path), path)
    unknown = sorted({p.id for p in demo.personas} - person_ids)
    if unknown:
        raise ConfigError(f"{path}: personas that are not people of the business: {unknown}")
    return demo
