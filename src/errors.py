class CallMeError(Exception):
    pass


class ParseError(CallMeError):
    pass


class GenerationError(CallMeError):
    pass