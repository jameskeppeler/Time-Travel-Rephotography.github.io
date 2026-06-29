"""Pure formatting helpers used by the rephotography GUI.

Extracted from MainWindow as part of the Sprint-4 module split. These
functions take only their arguments — no MainWindow state, no Qt — so
they're trivially unit-testable.
"""


def format_elapsed_for_summary(elapsed_seconds) -> str:
    """Render a seconds count as H:MM:SS / M:SS, or "N/A" for None.

    Negative inputs are clamped to 0 so a clock-skew or interrupted-run
    elapsed can never produce a negative time display.
    """
    if elapsed_seconds is None:
        return "N/A"
    elapsed = int(max(0, elapsed_seconds))
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def preflight_report_plain_text(report) -> str:
    """Render a preflight report dict as a plain-text summary suitable
    for the "Copy Report" button in the preflight dialog.
    """
    lines = [
        "Startup Preflight Report",
        f"Summary: {report['pass']} pass, {report['warn']} warn, {report['fail']} fail",
        "",
    ]
    for check in report["checks"]:
        lines.append(f"[{check['status'].upper()}] {check['name']}: {check['detail']}")
        if check.get("fix") and check["status"] != "pass":
            lines.append(f"  Fix: {check['fix']}")
    return "\n".join(lines)


def preflight_report_html(report) -> str:
    """Render a preflight report dict as the HTML body shown in the
    preflight dialog. Per-check status colors mirror the in-app log
    severity palette."""
    status_color = {"pass": "#6fcf97", "warn": "#f2c94c", "fail": "#eb5757"}
    rows = []
    for c in report["checks"]:
        color = status_color.get(c["status"], "#b7bcc5")
        fix_html = (
            f"<br/><span style='color:#b7bcc5'><b>Fix:</b> {c['fix']}</span>"
            if c.get("fix") and c["status"] != "pass"
            else ""
        )
        rows.append(
            f"<li><span style='color:{color}'><b>{c['status'].upper()}</b></span> "
            f"<b>{c['name']}</b>: {c['detail']}{fix_html}</li>"
        )
    return (
        f"<h3>Startup Preflight</h3>"
        f"<p><b>Summary:</b> {report['pass']} pass, {report['warn']} warn, {report['fail']} fail</p>"
        f"<ul>{''.join(rows)}</ul>"
    )


__all__ = [
    "format_elapsed_for_summary",
    "preflight_report_plain_text",
    "preflight_report_html",
]
