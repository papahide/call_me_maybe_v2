import argparse
import llm_sdk
import src.errors as errors
import src.parse as parse
import sys
import src.load_vocab as lvocab
import src.answer_gen as answer
import src.validation as val
from pydantic import ValidationError


def _add_default_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Adds the default paths for the json files:
    functions_definition, input prompts and output file
    (to be created if is not).

    Arg:
        parser: object in charge of managing the
                arguments for the program.
    Return:
        No return needed.
    """
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


def main() -> None:
    """
    Controls and executes all the main flux of the program:

    Parse the arguments.
    Creates an instance of the LLM model.
    Creates the vocabulary containing: ID -> token.
    Generates the output.
    Creates the output file.

    Catches all the errors that the main functions could raise.

    Args:
        No args needed.
    Return:
        No return needed.
    """
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
    try:
        model: llm_sdk.Small_LLM_Model = llm_sdk.Small_LLM_Model()
    except (OSError, RuntimeError, ValueError) as err:
        print(f"Failed to load LLM Model: {err}", file=sys.stderr)
        sys.exit(1)
    try:
        vocab: dict[int, str] = lvocab.create_vocab(model)
    except (errors.CallMeError, TypeError, ValueError, RuntimeError,
            AttributeError) as err:
        print(f"Error loading vocab file: {err}", file=sys.stderr)
        sys.exit(1)
    try:
        outputs: list[val.OutputFormat] = answer.output_generator(model,
                                                                  i_prom,
                                                                  f_def,
                                                                  vocab)
    except (errors.CallMeError, KeyError, StopIteration,
            ValidationError) as err:
        print(f"Error generating response: {err}", file=sys.stderr)
        sys.exit(1)
    try:
        parse.gen_output_file(outputs, args.output)
    except errors.CallMeError as err:
        print(f"Error generating output file: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
