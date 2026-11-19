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
                                                                float |
                                                                bool]]:
    selected_func: str = ans_name.complete_name(model, res, prompt, f_def, vocab)
    res.extend(procp.tokenize_str(model, '", "parameters": {'))
    parameters: dict[str,
                     str |
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
    for text in ['"replacement": "*"}}', '"replacement": "']:
        ids = procp.tokenize_str(model, text)
        print([vocab[i] for i in ids])
    results: list[val.OutputFormat] = []
    for p in i_prom:
        prompt: list[int] = procp.create_function_calling_prompt(model, f_def, p)
        res: list[int] = procp.create_forced_prefix(model, p)
        try:
            select_func, parameters = complete_response(model,
                                                        res,
                                                        prompt,
                                                        f_def,
                                                        vocab)
        except errors.CallMeError as err:
            raise errors.CallMeError(f"Error generating response: {err}") from err
        formated_result: val.OutputFormat = val.OutputFormat(
            prompt=p.prompt,
            name=select_func,
            parameters=parameters
        )
        results.append(formated_result)
    return results
