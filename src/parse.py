from pydantic import TypeAdapter, ValidationError
import src.validation as val
import src.errors as errors
from pathlib import Path
from typing import Any
import json


def validate_vocab(parsed: Any) -> dict[str, int]:
    """
    Validates the parsed vocabulary.

    Args:
        parsed: the parsed JSON content.

    Return:
        A dictionary containing tokens and their IDs.
    """
    try:
        adapter = TypeAdapter(dict[str, int])
        return adapter.validate_python(parsed)
    except ValidationError as err:
        raise errors.ParseError("Error trying to "
                                f"validate vocabulary: {err}") from err


def read_file(filepath: str) -> str:
    """
    Reads a file and raises errors if the file cannot be read.

    Args:
        filepath: the file path of the file to read.
    Return:
        A string with the content of the file.
    """
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
    """
    Parses the content of the json file passed as string to
    Python object and raises errors if fails.

    Args:
        file_content: the content of the json file received from "open()"
    Return:
        Returns the python object, any type of data.
    """
    try:
        return (json.loads(file_content))
    except json.JSONDecodeError as err:
        raise errors.ParseError(f"Error trying to parse json: "
                                f"{err}") from err


def validate_definitions(parsed: Any) -> list[val.FunctionDefinition]:
    """
    Uses TypeAdapter to serialize the parsed content received by json.loads.
    Organises the data in FunctionDefinition format.

    Args:
        parsed: receives the data from json.loads() and can be of any type.
    Return:
        Returns the formated data in a list of FunctionDefinition structure.
    """
    if parsed == []:
        raise errors.ParseError("Empty function definitions")
    try:
        adapter = TypeAdapter(list[val.FunctionDefinition])
        return adapter.validate_python(parsed)
    except ValidationError as err:
        raise errors.ParseError(f"Error trying to validate "
                                f"Function Definitions: {err}") from err


def validate_prompts(parsed: Any) -> list[val.PromptInput]:
    """
    Uses TypeAdapter to serialize the parsed content received by json.loads().
    Organises the data in PromptInput format.

    Args:
        parsed: receinces the data from json.loads() and can be of any type.
    Return:
        Returns the formated data in a list of PromptInput structure.
    """
    if parsed == []:
        raise errors.ParseError("Empty prompt input")
    try:
        adapter = TypeAdapter(list[val.PromptInput])
        return adapter.validate_python(parsed)
    except ValidationError as err:
        raise errors.ParseError(f"Error trying to validate "
                                f"Prompt inputs: {err}") from err


def parse_definitions(file: str) -> list[val.FunctionDefinition]:
    """
    Ensures that the function_definitions file is formatted correctly
    and saves the content in "FunctionDefinition" format using
    other functions.

    Args:
        file: the file path to the json file containing the
              functions_definitions.
    Return:
        A list with all the functions formatted in FunctionDefinition format.
    """
    json_content: Any = parse_json_file(read_file(file))
    return validate_definitions(json_content)


def parse_prompts(file: str) -> list[val.PromptInput]:
    """
    Ensures that the input prompts file is formatted correctly
    and saves the content in "PromptInput" format using
    other functions.

    Args:
        file: the file path to the json file containing the
              input prompts.
    Return:
        A list with all the functions formatted in PromptInput format.
    """
    json_content: Any = parse_json_file(read_file(file))
    return validate_prompts(json_content)


def gen_output_file(outputs: list[val.OutputFormat], output_path: str) -> None:
    """
    Serializes the json data from a OutputFormat format to json structure.

    Creates the father directory of the output_path that will contain the
    .json ouptut file.

    Creates the output file and raises errors in case there appears one.

    Args:
        outputs: a list of OutputFormat format data.
        output_path: path to create the output directory and file.
    Return:
        No returns.
    """
    adapter = TypeAdapter(list[val.OutputFormat])
    json_data = adapter.dump_json(outputs, indent=4,
                                  ensure_ascii=False).decode("utf-8")
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(json_data)
    except OSError as err:
        raise errors.CallMeError(f"Error writing output file: {err}") from err
