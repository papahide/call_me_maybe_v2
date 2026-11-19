from enum import Enum
from pydantic import BaseModel, ConfigDict


class ParameterType(str, Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"


class Parameter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: ParameterType


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter


class PromptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str


class OutputFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]
