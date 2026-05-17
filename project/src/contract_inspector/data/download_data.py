# src/contract_inspector/data/download_data.py

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import TypedDict

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True)
console = Console()


class SourceSpec(TypedDict):
    url: str
    archive: str
    extract_to: str
    md5: str | None


SOURCES: dict[str, SourceSpec] = {
    "cuad_full": {
        "url": "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1",
        "archive": "data/raw/cuad/CUAD_v1.zip",
        "extract_to": "data/raw/cuad/CUAD_v1",
        "md5": "c38f490a984420b8a62600db401fafd5",
    },
    "contractnli": {
        "url": "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip",
        "archive": "data/raw/contractnli/contract-nli.zip",
        "extract_to": "data/raw/contractnli/contract-nli",
        "md5": None,
    },
}


def run_command(command: list[str]) -> None:
    console.print(f"[dim]{' '.join(command)}[/dim]")
    subprocess.run(command, check=True)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


@app.command()
def download(
    source: str = typer.Argument(..., help="cuad_full | cuad_github_qa | contractnli | all"),
    force: bool = typer.Option(False),
) -> None:
    selected = list(SOURCES) if source == "all" else [source]

    for name in selected:
        if name not in SOURCES:
            raise typer.BadParameter(f"Unknown source: {name}")

        spec = SOURCES[name]
        archive_path = Path(spec["archive"])
        extract_to = Path(spec["extract_to"])

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        extract_to.parent.mkdir(parents=True, exist_ok=True)

        if archive_path.exists() and not force:
            console.print(f"[yellow]Skip download, exists: {archive_path}[/yellow]")
        else:
            run_command([
                "curl",
                "-L",
                "--fail",
                "--retry",
                "5",
                spec["url"],
                "-o",
                str(archive_path),
            ])

        expected_md5 = spec.get("md5")
        if expected_md5:
            actual_md5 = file_md5(archive_path)
            if actual_md5 != expected_md5:
                raise RuntimeError(
                    f"MD5 mismatch for {archive_path}: "
                    f"expected {expected_md5}, got {actual_md5}"
                )

        if extract_to.exists() and force:
            shutil.rmtree(extract_to)

        if extract_to.exists():
            console.print(f"[yellow]Skip unzip, exists: {extract_to}[/yellow]")
        else:
            extract_to.mkdir(parents=True, exist_ok=True)
            run_command(["unzip", "-q", str(archive_path), "-d", str(extract_to)])

        console.print(f"[green]Ready: {name} -> {extract_to}[/green]")


if __name__ == "__main__":
    app()