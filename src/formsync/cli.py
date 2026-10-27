"""Command-line entry point (``formsync ...``), built with Typer.

Step 0 provides only ``version`` and ``config``; later steps add ``analyze``, ``calibrate``,
``beats``, ``feedback`` and ``bench``. Each command is a thin wrapper that parses arguments and
delegates to library code, so all logic stays testable without the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from formsync import __version__
from formsync.config import get_settings, load_pipeline_config

app = typer.Typer(
    name="formsync",
    help="Dance formation & synchronization analyzer.",
    no_args_is_help=True,
    add_completion=False,
)
# highlight=False keeps output plain (no colour codes around numbers), which is friendlier for
# piping and for tests that assert on the text.
console = Console(highlight=False)


@app.command()
def version() -> None:
    """Print the installed formsync version."""
    console.print(f"formsync {__version__}")


@app.command()
def config(
    pipeline: Annotated[
        Path | None,
        typer.Option("--pipeline", "-p", help="Pipeline YAML to load (default: bundled config)."),
    ] = None,
) -> None:
    """Show the effective configuration: environment settings and pipeline constants.

    Secrets are masked. Useful to confirm which provider, model and thresholds a run will use.
    """
    settings = get_settings()
    cfg = load_pipeline_config(pipeline or settings.pipeline_config)
    console.print("[bold]Settings[/bold]")
    console.print_json(settings.model_dump_json(indent=2))
    console.print(f"[bold]Resolved LLM model:[/bold] {settings.resolved_llm_model()}")
    console.print("[bold]Pipeline config[/bold]")
    console.print_json(cfg.model_dump_json(indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
