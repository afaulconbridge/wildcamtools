"""Helpers for inferring a recording timestamp from a video filename.

The wildlife cameras in this project name files using a timestamp prefix
(e.g. ``20230816202116_VD_00001.MP4``).  This module exposes a small helper
that turns such a filename into a :class:`datetime.datetime` when the user
supplies a matching ``strftime`` pattern, and silently returns ``None`` when
no pattern is given or the filename does not match.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def infer_recorded_at(filename: str, fmt: str | None) -> datetime | None:
    """Infer a recording timestamp from ``filename`` using a ``strftime`` pattern.

    Only the basename of ``filename`` is considered.  The pattern is matched
    against the prefix of the basename, so a format like ``%Y%m%d%H%M%S``
    works for filenames such as ``20230816202116_VD_00001.MP4`` even though
    the full name contains trailing characters.

    Args:
        filename: The video filename (or absolute path; only the basename is used).
        fmt: A ``strftime``/``strptime`` pattern, or ``None`` to skip inference.

    Returns:
        The parsed :class:`datetime`, or ``None`` if ``fmt`` is falsy or the
        basename does not start with a string matching the pattern.

    """
    if not fmt:
        return None

    candidate = filename.rsplit("/", 1)[-1]
    min_len = _minimum_pattern_length(fmt)
    for end in range(len(candidate), min_len - 1, -1):
        try:
            return datetime.strptime(candidate[:end], fmt)
        except ValueError:
            continue
    logger.debug("Filename %r did not match format %r", candidate, fmt)
    return None


def _minimum_pattern_length(fmt: str) -> int:
    """Return the minimum number of characters required to satisfy ``fmt``.

    Used as a lower bound when scanning prefixes of a filename.  ``%Y`` and
    similar directives have a variable runtime width, so the real match may
    be longer than this; the search loop handles that.
    """
    i = 0
    length = 0
    while i < len(fmt):
        if fmt[i] == "%" and i + 1 < len(fmt):
            i += 2
        else:
            i += 1
            length += 1
    return length
