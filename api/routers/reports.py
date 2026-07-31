"""
Report Export Endpoints (Product Surface).

Provides export endpoints for downloading formatted reports (Weekly Review,
Advisor Recommendation Scorecard) as Text/Markdown/CSV formats.
"""

import csv
import io
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response, StreamingResponse

from agent.logger import log_to_component
from tools.weekly_review import build_weekly_review

router = APIRouter()

_SPREADSHEET_FORMULA_PREFIXES = ("=", "+", "-", "@")


def _csv_cell(value: object) -> object:
    """Return a value that spreadsheet apps cannot interpret as a formula."""
    if not isinstance(value, str):
        return value
    return f"'{value}" if value.lstrip().startswith(_SPREADSHEET_FORMULA_PREFIXES) else value


@router.get("/api/export/weekly-review")
def export_weekly_review(format: str = "markdown"):
    """
    Export the Weekly One-Page Review as a formatted Markdown or Text document.
    """
    try:
        review = build_weekly_review()
        generated_at = review.get("generated_at", "")
        period_label = (review.get("period") or {}).get("label", "")

        md_lines = [
            "# CairnIQ Weekly One-Page Review",
            f"**Period**: {period_label} | **Generated**: {generated_at}\n",
            "---",
        ]

        for sec in review.get("sections", []):
            title = sec.get("title", "Section")
            status = sec.get("status", "ok")
            note = sec.get("note", "")

            md_lines.append(f"\n## {title}")
            if note:
                md_lines.append(f"*{note}*")

            if status == "ok":
                for k, v in sec.items():
                    if k in ("key", "title", "status", "note", "roadmap"):
                        continue
                    label = k.replace('_', ' ').title()
                    if isinstance(v, list):
                        md_lines.append(f"- **{label}**:")
                        for item in v[:5]:
                            if isinstance(item, dict):
                                # Render dict items as "key: val | key: val" inline
                                parts = [f"{ik}: {iv}" for ik, iv in item.items() if iv is not None]
                                md_lines.append(f"  - {' | '.join(parts)}")
                            else:
                                md_lines.append(f"  - {item}")
                        if len(v) > 5:
                            md_lines.append(f"  - *… and {len(v) - 5} more*")
                    elif isinstance(v, dict):
                        md_lines.append(f"- **{label}**:")
                        for sub_k, sub_v in v.items():
                            md_lines.append(f"  - {sub_k.replace('_', ' ').title()}: {sub_v}")
                    else:
                        md_lines.append(f"- **{label}**: {v}")

        content = "\n".join(md_lines)
        filename = f"weekly_review_{generated_at[:10] if generated_at else 'latest'}.md"
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        log_to_component("server", "Reports", f"Export weekly review failed: {e}", level=logging.ERROR)
        return JSONResponse({"error": "Failed to export weekly review"}, status_code=500)


@router.get("/api/export/advisor-scorecard")
def export_advisor_scorecard(format: str = "csv"):
    """
    Export the Advisor Recommendation Scorecard history as CSV or JSON.
    """
    try:
        from tools.memory import load_memory
        memory = load_memory()
        recs = memory.get("past_recommendations", [])

        if format.lower() == "json":
            return JSONResponse({"count": len(recs), "recommendations": recs})

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Symbol", "Action", "Stated Date", "Entry Price", "Current/Exit Price", "Alpha Pct", "Status"])

        for r in recs:
            writer.writerow([_csv_cell(value) for value in (
                r.get("ticker") or r.get("symbol", ""),
                r.get("action", ""),
                r.get("stated_at") or r.get("date", ""),
                r.get("entry_price", ""),
                r.get("current_price") or r.get("exit_price", ""),
                r.get("alpha_pct", ""),
                r.get("status", ""),
            )])

        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=advisor_scorecard.csv"}
        )

    except Exception as e:
        log_to_component("server", "Reports", f"Export advisor scorecard failed: {e}", level=logging.ERROR)
        return JSONResponse({"error": "Failed to export advisor scorecard"}, status_code=500)
