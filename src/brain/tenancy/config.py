"""Business configuration: businesses/<id>/business.yaml, plus where its files live.

A business is configured by files only (no code change per business). The YAML is strict:
unknown fields are errors, so a typo cannot silently change behaviour.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=64)]
Language = Literal["es", "en"]


class ConfigError(ValueError):
    """A business file is missing or malformed."""


class Provider(StrEnum):
    """The wire protocol the answer model is called with (agentgate's provider names)."""

    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelConfig(_Strict):
    provider: Provider
    # Must match an entry in the policy's models section exactly.
    name: str = Field(min_length=1, max_length=128)
    # The provider origin the agentgate proxy forwards to. The brain never calls it directly.
    upstream: str = Field(pattern=r"^https?://[^\s/]+$")
    max_output_tokens: int = Field(ge=256, le=128_000)


class Business(_Strict):
    id: Slug
    name: str = Field(min_length=1, max_length=200)
    domains: tuple[str, ...] = Field(min_length=1)
    timezone: str
    default_language: Language
    model: ModelConfig

    @property
    def primary_domain(self) -> str:
        return self.domains[0]


@dataclass(frozen=True)
class BusinessPaths:
    """Where one business's configuration and runtime state live."""

    config: Path
    data: Path

    @property
    def business_file(self) -> Path:
        return self.config / "business.yaml"

    @property
    def people_file(self) -> Path:
        return self.config / "people.yaml"

    @property
    def items_dir(self) -> Path:
        return self.config / "items"

    @property
    def policy_dir(self) -> Path:
        return self.config / "policy"

    @property
    def rate_table(self) -> Path:
        return self.policy_dir / "rates.json"

    @property
    def evals_dir(self) -> Path:
        return self.config / "evals"


def business_paths(business_id: str, businesses_dir: Path, data_dir: Path) -> BusinessPaths:
    return BusinessPaths(config=businesses_dir / business_id, data=data_dir / business_id)


def read_yaml(path: Path) -> object:
    if not path.is_file():
        raise ConfigError(f"{path}: not found")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML ({exc.__class__.__name__})") from None


def validate[M: BaseModel](model: type[M], data: object, path: Path) -> M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or model.__name__}: {err['msg']}"
            for err in exc.errors(include_url=False, include_input=False)
        )
        raise ConfigError(f"{path}: {problems}") from None


def load_business(paths: BusinessPaths) -> Business:
    business = validate(Business, read_yaml(paths.business_file), paths.business_file)
    if business.id != paths.config.name:
        raise ConfigError(f"{paths.business_file}: id {business.id!r} does not match its folder")
    return business
