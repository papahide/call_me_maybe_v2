from pydantic import TypeAdapter, ValidationError
import src.validation as val
import src.errors as errors
from typing import Any
import json


def read_file(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            content: str = file.read()
            return content
    except UnicodeDecodeError as err:
        raise errors.ParseError(f"Error trying to encode file "
                                f"\"{filepath}\": {err}") from err
    except OSError as err:
        raise errors.ParseError(f"Error trying to open file "
                                f"\"{filepath}\": {err}") from err


def parse_json_file(file_content: str) -> Any:
    try:
        return (json.loads(file_content))
    except json.JSONDecodeError as err:
        raise errors.ParseError(f"Error trying to parse json: "
                                f"{err}") from err


def validate_definitions(parsed: Any) -> list[val.FunctionDefinition]:
    if parsed == []:
        raise errors.ParseError("Empty function definitions")
    try:
        adapter = TypeAdapter(list[val.FunctionDefinition])
        return adapter.validate_python(parsed)
    except ValidationError as err:
        raise errors.ParseError(f"Error trying to validate "
                                f"Function Definitions: {err}") from err


def validate_prompts(parsed: Any) -> list[val.PromptInput]:
    if parsed == []:
        raise errors.ParseError("Empty prompt input")
    try:
        adapter = TypeAdapter(list[val.PromptInput])
        return adapter.validate_python(parsed)
    except ValidationError as err:
        raise errors.ParseError(f"Error trying to validate "
                                f"Prompt inputs: {err}") from err


def parse_definitions(file: str) -> list[val.FunctionDefinition]:
    json_content: Any = parse_json_file(read_file(file))
    return validate_definitions(json_content)


def parse_prompts(file: str) -> list[val.PromptInput]:
    json_content: Any = parse_json_file(read_file(file))
    return validate_prompts(json_content)
