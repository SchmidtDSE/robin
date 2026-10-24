"""ANSI color codes for human output. `main` decides when color is on."""

RED_BOLD = "\x1b[1;31m"
YELLOW = "\x1b[33m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
RESET = "\x1b[0m"

STATUS_COLORS = {
    "finished": GREEN,
    "finished_with_failures": RED,
    "preparation_failed": RED,
    "preparing": YELLOW,
    "queued": YELLOW,
    "running": YELLOW,
    "cancelled": DIM,
    "withdrawn": DIM,
}


def add_color(text: str, color: str | None, *, enabled: bool) -> str:
    """`text` in `color`, or `text` unchanged if color is off or `color` is None."""
    if not enabled or color is None:
        return text
    return f"{color}{text}{RESET}"
