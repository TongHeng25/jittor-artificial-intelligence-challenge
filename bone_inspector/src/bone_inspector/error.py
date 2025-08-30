from typing import Optional, Union, Tuple, List, Callable

import inspect

def _default_formatter(code: str, message: str, file: str, line: str) -> str:
    return f"[{code}] from {file}, line {line}: {message}"

class ExtractError(Exception):
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

def error_unknown(**kwargs) -> ExtractError:
    return ExtractError(
        code='ERROR_UNKNOWN',
        message="Unknown error.",
        go_back=2,
        **kwargs,
    )

def error_bad_topoplgy(**kwargs) -> ExtractError:
    return ExtractError(
        code='ERROR_BAD_TOPOLOGY',
        message="Bad topology found.",
        go_back=2,
        **kwargs,
    )

def error_file_does_not_exist(path: Optional[str]=None, **kwargs) -> ExtractError:
    return ExtractError(
        code='ERROR_FILE_DOES_NOT_EXIST',
        message=f"File does not exist: {path}" if path is not None else "File does not exist.",
        go_back=2,
        **kwargs,
    )

def error_unsupported_type(type: Optional[str]=None, **kwargs) -> ExtractError:
    return ExtractError(
        code='ERROR_UNSUPPORTED_TYPE',
        message=f"Unsupported type: {type}" if type is not None else "Unsupported type.",
        go_back=2,
        **kwargs,
    )

def error_incorrect_vrm_addon(**kwargs) -> ExtractError:
    return ExtractError(
        code='ERROR_INCORRECT_VRM_ADDON',
        message="VRM addon is missing or incorrect version is installed.",
        go_back=2,
        **kwargs,
    )

def error_no_armature(**kwargs) -> ExtractError:
    return ExtractError(
        code='ERROR_NO_ARMATRUE',
        message='No armature found in the scene.',
        go_back=2,
        **kwargs,
    )

def error_multiple_armatrues(**kwargs) -> ExtractError:
    return ExtractError(
        code='ERROR_MULTIPLE_ARMATURES',
        message='Multiple armatures found in the scene.',
        go_back=2,
        **kwargs,
    )

def error_no_track(**kwargs) -> ExtractError:
    return ExtractError(
        code='ERROR_NO_TRACK',
        message='No track found in the scene.',
        go_back=2,
        **kwargs,
    )