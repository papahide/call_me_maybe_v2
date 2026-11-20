from enum import Enum
from pydantic import BaseModel, ConfigDict, field_validator


class ParameterType(str, Enum):
    """Defines the supported parameter types."""
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    INTEGER = "integer"


class Parameter(BaseModel):
    """Defines the schema of a function parameter."""
    model_config = ConfigDict(extra="forbid")
    type: ParameterType


class FunctionDefinition(BaseModel):
    """Defines a function and its parameters."""
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter

    @field_validator("name", "description")
    @classmethod
    def check_not_blank(cls, value: str) -> str:
        """
        Reject empty or whitespace-only text.
        """
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("parameters")
    @classmethod
    def check_param_names(
        cls, value: dict[str, Parameter]
    ) -> dict[str, Parameter]:
        """
        Reject empty or whitespace-only parameter names.
        """
        if any(not key.strip() for key in value):
            raise ValueError("parameter names must not be empty")
        return value


class PromptInput(BaseModel):
    """Defines an input prompt."""
    model_config = ConfigDict(extra="forbid")
    prompt: str

    @field_validator("prompt")
    @classmethod
    def check_not_blank(cls, value: str) -> str:
        """
        Reject empty or whitespace-only prompts.
        """
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class OutputFormat(BaseModel):
    """Defines the structure of a generated function call."""
    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]
