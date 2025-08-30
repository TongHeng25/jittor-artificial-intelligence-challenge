from typing import Optional, Union, Tuple, List, Callable

import inspect

def _default_formatter(code: str, message: str, file: str, line: str) -> str:
    return f"[{code}] from {file}, line {line}: {message}"

class ExtractWarning():
    code: str
    message: str
    
    def __init__(
        self,
        code: str,
        message: str,
        file: Optional[str]=None,
        line: Optional[str]=None,
        formatter: Callable[[str, str, str, int], str]=_default_formatter,
        go_back: int=1,
    ):
        if file is None or line is None:
            frame = inspect.currentframe().f_back
            for i in range(go_back-1):
                frame = frame.f_back
            file = file or frame.f_code.co_filename
            line = line or frame.f_lineno
        self.code = code
        self.message = formatter(code, message, file, line)
        self.file = file
        self.line = line

def warning_multiple_tracks(**kwargs) -> ExtractWarning:
    return ExtractWarning(
        code='WARNING_MULTIPLE_TRACKS',
        message='Multiple tracks found.',
        **kwargs,
    )

def warning_no_skin(**kwargs) -> ExtractWarning:
    return ExtractWarning(
        code='WARNING_NO_SKIN',
        message='No skin found for certain vertex.',
        **kwargs,
    )