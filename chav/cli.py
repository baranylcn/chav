from __future__ import annotations

import argparse
import sys

from chav.engine import analyze


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chav",
        description="Chav — diagnostic engine for tabular datasets",
    )
    sub = parser.add_subparsers(dest="command")

    analyze_cmd = sub.add_parser("analyze", help="Run diagnostics on a CSV file")
    analyze_cmd.add_argument("data", help="Path to the input CSV file")
    analyze_cmd.add_argument("--reference", metavar="FILE", help="Path to a reference CSV file")
    analyze_cmd.add_argument("--target", metavar="COL", help="Target column name")
    analyze_cmd.add_argument("--time-column", metavar="COL", help="Datetime column name")
    analyze_cmd.add_argument(
        "--format",
        choices=["summary", "json", "csv"],
        default="summary",
        help="Output format (default: summary)",
    )
    analyze_cmd.add_argument("--all", action="store_true", help="Include passing diagnostics in output")
    analyze_cmd.add_argument("--output-file", metavar="FILE", help="Write output to file instead of stdout")

    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        return 1

    try:
        report = analyze(
            data=parsed.data,
            reference_data=parsed.reference,
            target=parsed.target,
            time_column=parsed.time_column,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    include_all: bool = parsed.all

    if parsed.format == "json":
        output = report.to_json(all=include_all)
    elif parsed.format == "csv":
        output = report.to_csv(all=include_all) or ""
    else:
        output = report.summary()

    if parsed.output_file:
        with open(parsed.output_file, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        print(output)

    return 0


def cli_entry() -> None:
    sys.exit(main())
