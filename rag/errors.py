"""User-facing errors. The message of each is safe to show in the UI."""


class GiltoError(Exception):
    """Base class. str(error) is a plain-language message for the user."""


class LoaderError(GiltoError):
    pass


class LimitError(GiltoError):
    pass


class ConfigError(GiltoError):
    pass


class LLMError(GiltoError):
    pass
