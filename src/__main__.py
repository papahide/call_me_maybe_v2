import argparse
import llm_sdk
import src.errors as errors
import src.parse as parse
import sys
import src.load_vocab as lvocab
import src.answer_gen as answer
import src.validation as val
from pydantic import TypeAdapter, ValidationError
from pathlib import Path


def _add_default_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to JSON file with function definitions")

    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to JSON file with input tests")

    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to JSON file with output result")


def gen_output_file(outputs: list[val.OutputFormat], output_path: str) -> None:
    adapter = TypeAdapter(list[val.OutputFormat])
    json_data = adapter.dump_json(outputs, indent=4, ensure_ascii=False).decode("utf-8")
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(json_data)
    except OSError as err:
        raise errors.CallMeError(f"Error writing output file: {err}") from err


def main() -> None:
    """Parseo"""
    parser = argparse.ArgumentParser(
        description="Process file paths")
    _add_default_arguments(parser)
    args: argparse.Namespace = parser.parse_args()
    try:
        f_def = parse.parse_definitions(args.functions_definition)
        i_prom = parse.parse_prompts(args.input)
    except errors.CallMeError as err:
        print(err, file=sys.stderr)
        sys.exit(1)
    """Cargar el modelo"""
    try:
        model: llm_sdk.Small_LLM_Model = llm_sdk.Small_LLM_Model()
    except (OSError, RuntimeError, ValueError) as err:
        print(f"Failed to load LLM Model: {err}", file=sys.stderr)
        sys.exit(1)
    """Cargar el vocabulario"""
    try:
        vocab: dict[int, str] = lvocab.create_vocab(model)
    except (errors.CallMeError, TypeError, ValueError, RuntimeError,
            AttributeError) as err:
        print(f"Error loading vocab file: {err}", file=sys.stderr)
        sys.exit(1)
    """Hablar con modelo y generar json"""
    try:
        outputs: list[val.OutputFormat] = answer.output_generator(model,
                                                                  i_prom,
                                                                  f_def,
                                                                  vocab)
    except (errors.CallMeError, KeyError, StopIteration, ValidationError) as err:
        print(f"Error generating response: {err}", file=sys.stderr)
        sys.exit(1)
    """Crear el fichero de salida con json"""
    try:
        gen_output_file(outputs, args.output)
    except errors.CallMeError as err:
        print(f"Error generating output file: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
