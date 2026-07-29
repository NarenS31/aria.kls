#!/usr/bin/env python3.11
"""Launch the local ARIA demo on the first available port."""

from __future__ import annotations

import socket

from ui.app import CSS, create_demo


def find_free_port(start: int = 7860) -> int:
    for port in range(start, start + 10):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found between {start} and {start + 9}")


def main() -> None:
    demo = create_demo()
    port = find_free_port()
    print(f"ARIA is running at http://127.0.0.1:{port}")
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        footer_links=[],
        quiet=True,
        css=CSS,
    )


if __name__ == "__main__":
    main()
