#!/usr/bin/env python3
"""Render PRIMA prediction JSON into human-readable Markdown and HTML."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PRIORITY_VI = {"high": "Ưu tiên cao", "low": "Ưu tiên thấp", "none": "Không ưu tiên"}
PRIORITY_EN = {"high": "High priority", "low": "Low priority", "none": "No priority"}

LABELS_VI = {
    "cerebral_atrophy": "Teo não",
    "intracranial_hemorrhage": "Xuất huyết nội sọ",
    "large_vessel_stroke": "Nhồi máu mạch lớn",
    "cerebral_aneurysm": "Phình động mạch não",
    "arteriovenous_malformation": "Dị dạng động-tĩnh mạch",
    "moyamoya": "Bệnh Moyamoya",
    "adult_glioma": "U thần kinh đệm người lớn",
}


def scalar(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    flat = value
    while isinstance(flat, list) and len(flat) == 1:
        flat = flat[0]
    if isinstance(flat, (int, float)):
        return float(flat)
    raise ValueError(f"Expected scalar-like prediction, got {type(value).__name__}")


def normalize_code(code: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", code.lower()).strip("_")


def humanize_code(code: str) -> str:
    text = re.sub(r"[_\-]+", " ", code).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:1].upper() + text[1:] if text else code


def label_for(code: str, language: str) -> str:
    normalized = normalize_code(code)
    if language == "vi":
        if normalized in LABELS_VI:
            return LABELS_VI[normalized]
        for key, translated in LABELS_VI.items():
            if normalized.endswith("_" + key):
                return translated
    return humanize_code(code)


def sorted_scores(group: Dict[str, Any]) -> List[Tuple[str, float]]:
    return sorted(
        [(name, scalar(value)) for name, value in group.items()],
        key=lambda item: item[1],
        reverse=True,
    )


def classify_threshold_group(group: Dict[str, Any], near_margin: float):
    rows = sorted_scores(group)
    positive = [(n, s) for n, s in rows if s >= 0]
    near_negative = [(n, s) for n, s in rows if -near_margin <= s < 0]
    negative = [(n, s) for n, s in rows if s < -near_margin]
    return positive, near_negative, negative


def priority_decision(priority: Dict[str, Any]):
    rows = sorted_scores(priority)
    return rows[0] if rows else None


def fmt_margin(value: float) -> str:
    return f"{value:+.3f}"


def technical_summary(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        return {}
    stages = metrics.get("stages", {})
    return {
        "series_count": len(metrics.get("series", [])),
        "skipped_series_count": len(metrics.get("skipped_series", [])),
        "total_seconds": stages.get("total", {}).get("seconds"),
        "tokenizer_seconds": stages.get("tokenizer_total", {}).get("seconds"),
        "prima_seconds": stages.get("prima_inference", {}).get("seconds"),
        "gpu_name": metrics.get("gpu_name"),
        "tokenizer_chunk_limit": metrics.get("tokenizer_runtime_chunk_limit"),
    }


def build_markdown(
    predictions: Dict[str, Any],
    *,
    study_id: str,
    metrics: Optional[Dict[str, Any]] = None,
    language: str = "vi",
    near_margin: float = 0.25,
    show_all_scores: bool = False,
) -> str:
    if language not in {"vi", "en"}:
        raise ValueError("language must be 'vi' or 'en'")
    if near_margin < 0:
        raise ValueError("near_margin must be >= 0")

    diagnosis = predictions.get("diagnosis", {})
    referral = predictions.get("referral", {})
    priority = predictions.get("priority", {})
    d_pos, d_near, d_neg = classify_threshold_group(diagnosis, near_margin)
    r_pos, r_near, r_neg = classify_threshold_group(referral, near_margin)
    p_decision = priority_decision(priority)
    tech = technical_summary(metrics)

    if language == "vi":
        lines = [
            f"# PRIMA MRI Brain — {study_id}",
            "",
            "> **Lưu ý:** Đây là đầu ra hỗ trợ nghiên cứu của PRIMA, không phải chẩn đoán xác định. "
            "Các giá trị là **margin so với ngưỡng mô hình**, không phải phần trăm xác suất bệnh.",
            "",
            "## Mức ưu tiên",
            "",
        ]
        if p_decision:
            p_name, p_score = p_decision
            lines.append(
                f"**{PRIORITY_VI.get(normalize_code(p_name), label_for(p_name, language))}** "
                f"(score {fmt_margin(p_score)})"
            )
        else:
            lines.append("_Không có đầu ra priority._")

        lines += ["", "## Diagnosis — đầu ra vượt ngưỡng", ""]
        if d_pos:
            for name, score in d_pos:
                lines.append(
                    f"- **{label_for(name, language)}** — margin `{fmt_margin(score)}`  "
                    f"<small>({name})</small>"
                )
        else:
            lines.append("_Không có diagnosis output nào vượt ngưỡng 0._")

        lines += ["", f"## Diagnosis — gần ngưỡng âm (0 đến -{near_margin:g})", ""]
        if d_near:
            for name, score in d_near:
                lines.append(
                    f"- {label_for(name, language)} — margin `{fmt_margin(score)}`  "
                    f"<small>({name})</small>"
                )
        else:
            lines.append("_Không có._")

        lines += ["", "## Referral — đầu ra vượt ngưỡng", ""]
        if r_pos:
            for name, score in r_pos:
                lines.append(
                    f"- **{label_for(name, language)}** — margin `{fmt_margin(score)}`  "
                    f"<small>({name})</small>"
                )
        else:
            lines.append("_Không có referral output nào vượt ngưỡng 0._")

        if r_near:
            lines += ["", f"### Referral gần ngưỡng âm (0 đến -{near_margin:g})", ""]
            for name, score in r_near:
                lines.append(
                    f"- {label_for(name, language)} — margin `{fmt_margin(score)}`  "
                    f"<small>({name})</small>"
                )

        if show_all_scores:
            lines += ["", "## Toàn bộ score", "", "### Diagnosis", ""]
            for name, score in d_pos + d_near + d_neg:
                lines.append(f"- {name}: `{fmt_margin(score)}`")
            lines += ["", "### Referral", ""]
            for name, score in r_pos + r_near + r_neg:
                lines.append(f"- {name}: `{fmt_margin(score)}`")
            lines += ["", "### Priority", ""]
            for name, score in sorted_scores(priority):
                lines.append(f"- {name}: `{fmt_margin(score)}`")

        lines += ["", "## Thông tin kỹ thuật", ""]
        if tech:
            lines.append(f"- Series xử lý: **{tech['series_count']}**")
            lines.append(f"- Series lỗi/bị bỏ: **{tech['skipped_series_count']}**")
            if tech.get("total_seconds") is not None:
                lines.append(f"- Tổng thời gian: **{tech['total_seconds']:.1f} giây**")
            if tech.get("tokenizer_seconds") is not None:
                lines.append(f"- Tokenizer: {tech['tokenizer_seconds']:.1f} giây")
            if tech.get("prima_seconds") is not None:
                lines.append(f"- PRIMA inference: {tech['prima_seconds']:.1f} giây")
            if tech.get("gpu_name"):
                lines.append(f"- GPU: {tech['gpu_name']}")
            if tech.get("tokenizer_chunk_limit") is not None:
                lines.append(f"- Tokenizer chunk limit: {tech['tokenizer_chunk_limit']}")
        else:
            lines.append("_Không co runtime metrics._")

        lines += [
            "",
            "---",
            "",
            "Raw machine output được giữ riêng trong file `*_predictions.json`. "
            "CLIP embedding không hiển thị trong báo cáo này.",
        ]
        return "\n".join(lines) + "\n"

    lines = [
        f"# PRIMA MRI Brain — {study_id}",
        "",
        "> **Note:** Research decision-support output only. Scores are model margins "
        "relative to task thresholds, not calibrated disease probabilities.",
        "",
        "## Priority",
        "",
    ]
    if p_decision:
        p_name, p_score = p_decision
        lines.append(
            f"** {PRIORITY_EN.get(normalize_code(p_name), label_for(p_name, language))}** "
            f"(score {fmt_margin(p_score)})"
        )
    else:
        lines.append("_No priority output._")
    lines += ["", "## Diagnosis outputs above threshold", ""]
    for name, score in d_pos:
        lines.append(f"- **{label_for(name, language)}** — `{fmt_margin(score)}` ({name})")
    if not d_pos:
        lines.append("_None._")
    lines += ["", "## Referral outputs above threshold", ""]
    for name, score in r_pos:
        lines.append(f"- **{label_for(name, language)}** — `{fmt_margin(score)}` ({name})")
    if not r_pos:
        lines.append("_None._")
    return "\n".join(lines) + "\n"


def markdown_to_html(markdown_text: str, *, study_id: str) -> str:
    body: List[str] = []
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            body.append("</ul>")
            in_list = False

    for raw in markdown_text.splitlines():
        line = raw.strip()
        if not line:
            close_list()
            continue
        if line == "---":
            close_list()
            body.append("<hr>")
        elif line.startswith("# "):
            close_list()
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            close_list()
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            close_list()
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("> "):
            close_list()
            body.append(f'<div class="notice">{html.escape(line[2:].replace("**", ""))}</div>')
        elif line.startswith("- "):
            if not in_list:
                body.append("<ul>")
                in_list = True
            item = html.escape(line[2:])
            item = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", item)
            item = re.sub(r"`(.+?)`", r"<code>\1</code>", item)
            item = re.sub(r"&lt;small&gt;(.+?)&lt;/small&gt;", r"<small>\1</small>", item)
            body.append(f"<li>{item}</li>")
        else:
            close_list()
            if line.startswith("_") and line.endswith("_"):
                body.append(f"<p><em>{html.escape(line[1:-1])}</em></p>")
            else:
                text = html.escape(line)
                text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
                text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
                body.append(f"<p>{text}</p>")
    close_list()

    return f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PRIMA — {html.escape(study_id)}</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: system-ui,-apple-system,"Segoe UI",sans-serif; margin:0; line-height:1.55; background:#f5f6f8; color:#202124; }}
main {{ max-width:900px; margin:32px auto; background:white; padding:36px 42px; border-radius:16px; box-shadow:0 4px 24px rgba(0,0,0,.08); }}
h1 {{ margin-top:0; font-size:28px; }}
h2 {{ margin-top:30px; border-bottom:1px solid #e7e7e7; padding-bottom:8px; }}
.notice {{ padding:14px 16px; border-left:4px solid #777; background:#f3f3f3; border-radius:6px; }}
li {{ margin:8px 0; }}
code {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; background:#f0f1f2; padding:2px 6px; border-radius:5px; }}
small {{ opacity:.7; }}
@media (prefers-color-scheme: dark) {{
  body {{ background:#15171a; color:#e8eaed; }}
  main {{ background:#202124; box-shadow:none; }}
  h2 {{ border-color:#3c4043; }}
  .notice {{ background:#2b2d31; }}
  code {{ background:#303134; }}
}}
@media print {{
  body {{ background:white; }}
  main {{ max-width:none; margin:0; box-shadow:none; padding:0; }}
}}
</style>
</head><body><main>{''.join(body)}</main></body></html>
"""


def render_reports(
    predictions: Dict[str, Any],
    *,
    study_id: str,
    output_dir: Path,
    metrics: Optional[Dict[str, Any]] = None,
    language: str = "vi",
    near_margin: float = 0.25,
    show_all_scores: bool = False,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown(
        predictions,
        study_id=study_id,
        metrics=metrics,
        language=language,
        near_margin=near_margin,
        show_all_scores=show_all_scores,
    )
    md_path = output_dir / f"{study_id}_report.md"
    html_path = output_dir / f"{study_id}_report.html"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(markdown_to_html(markdown, study_id=study_id), encoding="utf-8")
    return md_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PRIMA human-readable report")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--study-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--language", choices=["vi", "en"], default="vi")
    parser.add_argument("--near-margin", type=float, default=0.25)
    parser.add_argument("--show-all-scores", action="store_true")
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    metrics = (
        json.loads(args.metrics.read_text(encoding="utf-8"))
        if args.metrics and args.metrics.exists()
        else None
    )
    study_id = args.study_id or (metrics or {}).get("study_id") or args.predictions.name.removesuffix("_predictions.json")
    output_dir = args.output_dir or args.predictions.parent

    md_path, html_path = render_reports(
        predictions,
        study_id=study_id,
        output_dir=output_dir,
        metrics=metrics,
        language=args.language,
        near_margin=args.near_margin,
        show_all_scores=args.show_all_scores,
    )
    print(md_path)
    print(html_path)


if __name__ == "__main__":
    main()
