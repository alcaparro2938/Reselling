"""photofix -- straighten and reframe product photos for marketplace listings.

    from photofix import Config, process_file
    result = process_file("IMG_1234.HEIC", Config(ratio="4:5"))
    result.image        # the finished BGR image
    result.report       # what was done, and why
"""

from .config import Config, DEFAULT_RATIOS, RATIO_ALIASES, resolve_ratio
from .pipeline import Report, Result, debug_overlay, process, process_file

__version__ = "1.0.0"

__all__ = [
    "Config",
    "DEFAULT_RATIOS",
    "RATIO_ALIASES",
    "Report",
    "Result",
    "debug_overlay",
    "process",
    "process_file",
    "resolve_ratio",
    "__version__",
]
