import os
import sys
import time
from contextlib import contextmanager


_PROFILE_ENABLED = os.environ.get("MM_PROFILE", "0") not in ("0", "false", "False", None)
_PROFILE_LOG_PATH = os.environ.get("MM_PROFILE_LOG")  # e.g., "./timings.log"

_stdout_orig = sys.stdout
_stderr_orig = sys.stderr
_log_file_handle = None


def set_profile_enabled(enabled: bool) -> None:
    global _PROFILE_ENABLED
    _PROFILE_ENABLED = bool(enabled)

def set_profile_log_file(path: str) -> None:
    """
    Sets a file where timing lines will also be appended.
    This does NOT redirect all stdout; use enable_stdout_log for that.
    """
    global _PROFILE_LOG_PATH
    _PROFILE_LOG_PATH = path

def _emit(line: str):
    # Always print to current stdout
    print(line)
    # Optionally append to timing log file
    if _PROFILE_LOG_PATH:
        try:
            with open(_PROFILE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass  # fail-soft on file writing

def is_profile_enabled() -> bool:
    return _PROFILE_ENABLED

class _TeeStream:
    """
    Tee for stdout/stderr: writes to original stream and a file handle.
    """
    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, s):
        try:
            self._stream.write(s)
        except Exception:
            pass
        try:
            self._fh.write(s)
        except Exception:
            pass

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._fh.flush()
        except Exception:
            pass

def enable_stdout_log(path: str, mode: str = "a", also_stderr: bool = True) -> None:
    """
    Redirects all stdout (and optionally stderr) to both console and file `path`.
    Safe to call once; calling again reopens the file and resets tee.
    """
    global _log_file_handle, _stdout_orig, _stderr_orig
    # Close previous handle if any
    try:
        if _log_file_handle is not None and not _log_file_handle.closed:
            _log_file_handle.close()
    except Exception:
        pass
    # Open new file handle
    _log_file_handle = open(path, mode, encoding="utf-8")
    sys.stdout = _TeeStream(_stdout_orig, _log_file_handle)
    if also_stderr:
        sys.stderr = _TeeStream(_stderr_orig, _log_file_handle)

def disable_stdout_log() -> None:
    """
    Restores original stdout/stderr and closes log file handle.
    """
    global _log_file_handle, _stdout_orig, _stderr_orig
    sys.stdout = _stdout_orig
    sys.stderr = _stderr_orig
    try:
        if _log_file_handle is not None and not _log_file_handle.closed:
            _log_file_handle.close()
    except Exception:
        pass
    _log_file_handle = None

@contextmanager
def timer(tag: str, enabled: bool = None):
    """
    Lightweight timing context manager shared across modules.
    Prints wall-clock seconds when enabled.

    Enable precedence:
    - explicit `enabled` arg if provided
    - environment variable MM_PROFILE
    - default False
    """
    global _PROFILE_ENABLED
    do_print = _PROFILE_ENABLED if enabled is None else bool(enabled)
    if not do_print:
        yield
        return
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        _emit(f"[TIMING] {tag}: {dt:.4f}s")

def time_block(tag: str, enabled: bool = None):
    """Alias to use as a function in code: with time_block('tag'):"""
    return timer(tag, enabled)