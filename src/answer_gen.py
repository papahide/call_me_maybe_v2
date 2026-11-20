import llm_sdk
import src.process_prompt as procp
import src.validation as val
import src.errors as errors
import src.answer_name_utils as ans_name
import src.answer_parameters_utils as ans_p


def complete_response(model: llm_sdk.Small_LLM_Model,
                      res: list[int],
                      prompt: list[int],
                      f_def: list[val.FunctionDefinition],
                      vocab: dict[int, str]) -> tuple[str, dict[str,
                                                                str |
                                                                int |
                                                                float |
                                                                bool]]:
    """
    Completes the function calling response by generating the
    selected function name and its parameters.

    Uses complete_name() to generate the function name and
    complete_parameters() to generate its arguments through
    constrained decoding.

    Args:
        model: the language model used to generate the response.
        res: the token IDs generated so far.
        prompt: the initial prompt token IDs.
        f_def: a list containing the available function definitions.
        vocab: a dictionary mapping token IDs to their string values.

    Return:
        A tuple containing the selected function name and a
        dictionary with its generated parameters.
    """
    selected_func: str = ans_name.complete_name(model, res,
                                                prompt, f_def,
                                                vocab)
    res.extend(procp.tokenize_str(model, '", "parameters": {'))
    parameters: dict[str,
                     str |
                     int |
                     float |
                     bool] = ans_p.complete_parameters(model, res, prompt,
                                                       f_def,
                                                       selected_func,
                                                       vocab)
    return selected_func, parameters


def output_generator(model: llm_sdk.Small_LLM_Model,
                     i_prom: list[val.PromptInput],
                     f_def: list[val.FunctionDefinition],
                     vocab: dict[int, str]) -> list[val.OutputFormat]:

    """
    Generates the function calling responses for all input prompts
    and formats the results as OutputFormat objects.

    Creates the function calling prompt and forced prefix for each
    input prompt, then uses complete_response() to generate the
    selected function and its parameters.

    Raises an error if a problem occurs during response generation.

    Args:
        model: an instance of the LLM model.
        i_prom: a list containing the input prompts.
        f_def: a list containing the available function definitions.
        vocab: a dictionary mapping token IDs to their string values.

    Return:
        A list containing the generated responses in OutputFormat format.
    """
    results: list[val.OutputFormat] = []
    for p in i_prom:
        prompt: list[int] = procp.create_function_calling_prompt(model,
                                                                 f_def,
                                                                 p)
        res: list[int] = procp.create_forced_prefix(model, p)
        try:
            select_func, parameters = complete_response(model,
                                                        res,
                                                        prompt,
                                                        f_def,
                                                        vocab)
        except errors.CallMeError as err:
            raise errors.CallMeError("Error generating response: "
                                     f"{err}") from err
        formated_result: val.OutputFormat = val.OutputFormat(
            prompt=p.prompt,
            name=select_func,
            parameters=parameters
        )
        results.append(formated_result)
    return results
