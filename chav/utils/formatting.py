from __future__ import annotations

from chav.typing import Diagnostic, Status, Severity


def format_diagnostic(d: Diagnostic) -> str:
    cols = ", ".join(d.affected_columns) if d.affected_columns else "-"
    return f"  [{d.status.value.upper():4s}] {d.rule:<25s}  severity={d.severity.value:<6s}  confidence={d.confidence:.2f}  columns=[{cols}]"


def format_summary(diagnostics: list[Diagnostic], rows: int, columns: int) -> str:
    counts = {"pass": 0, "warn": 0, "fail": 0, "skipped": 0, "error": 0}
    for d in diagnostics:
        counts[d.status.value] = counts.get(d.status.value, 0) + 1

    header = (
        f"Chav Report | {rows} rows x {columns} cols | "
        f"{counts['fail']} fail, {counts['warn']} warn, "
        f"{counts['pass']} pass, {counts['skipped']} skipped"
    )

    lines = [header, "-" * len(header)]

    for d in diagnostics:
        if d.status in (Status.FAIL, Status.WARN):
            lines.append(format_diagnostic(d))

    if counts["pass"] + counts["skipped"] > 0:
        lines.append(f"  ... {counts['pass']} passed, {counts['skipped']} skipped")

    return "\n".join(lines)
