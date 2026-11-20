class CallMeError(Exception):
    """Base exception for the program."""
    pass


class ParseError(CallMeError):
    """Exception raised during parsing."""
    pass


class GenerationError(CallMeError):
    """Exception raised during generation."""
    pass
