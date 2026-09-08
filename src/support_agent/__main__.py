"""Run the HTTP service with uvicorn: ``python -m support_agent``.

This is the container entry point. For local development prefer
``support-agent serve``, which binds the loopback interface by default.
"""

from __future__ import annotations

import os
from typing import Final

import uvicorn

from support_agent.config import get_settings

#: Interface to bind. A container's published port controls reachability, and a
#: process bound to 127.0.0.1 inside a container is unreachable from outside it,
#: so binding every interface is the correct default here. Overridable for
#: anyone running this module directly on a host.
DEFAULT_HOST: Final[str] = "0.0.0.0"  # noqa: S104  # nosec B104 - see above
DEFAULT_PORT: Final[int] = 8000


def main() -> None:
    """Start the ASGI server using the configured log level."""
    settings = get_settings()
    uvicorn.run(
        "support_agent.api.app:create_app",
        factory=True,
        host=os.environ.get("AGENT_BIND_HOST", DEFAULT_HOST),
        port=int(os.environ.get("AGENT_BIND_PORT", DEFAULT_PORT)),
        log_level=settings.observability.log_level.lower(),
        # The application emits its own structured access log with correlation
        # and redaction; uvicorn's would be a second, unredacted format.
        access_log=False,
        server_header=False,
        date_header=True,
    )


if __name__ == "__main__":
    main()
