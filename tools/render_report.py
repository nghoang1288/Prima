#!/usr/bin/env python3
"""Render PRIMA prediction JSON into human-readable Markdown and HTML."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tools.evidence import build_evidence_html, generate_study_evidence
except ImportError:
    try:
        from evidence import build_evidence_html, generate_study_evidence
    except ImportError:
        build_evidence_html = None
        generate_study_evidence = None

PRIORITY_VI = {"high": "Ưu tiên cao", "low": "Ưu tiên thấp", "none": "Không ưu tiên"}
PRIORITY_EN = {"high": "High priority", "low": "Low priority", "none": "No priority"}

DIAGNOSIS_LABELS_VI = {
    "cyst_arachnoid_cyst": "Nang màng nhện",
    "cyst_colloid_cyst": "Nang keo",
    "developmental_dandy_walker_malformation": "Dị dạng Dandy-Walker",
    "developmental_dysgenesis_corpus_callosum": "Bất thường phát triển thể chai",
    "developmental_heterotopia": "Dị tật lạc chỗ (heterotopia)",
    "infectious_brain_abscess": "Áp xe não",
    "infectious_viral_encephalitis": "Viêm não do virus",
    "inflammatory_multiple_sclerosis": "Đa xơ cứng",
    "inflammatory_neurosarcoid": "Sarcoidosis thần kinh",
    "sellar_craniopharyngioma": "U sọ hầu",
    "sellar_pituitary_adenoma": "U tuyến yên",
    "sellar_rathkes_cleft_cyst": "Nang khe Rathke",
    "spine_syrinx": "Rỗng tủy",
    "structural_brain_herniation": "Thoát vị não",
    "structural_cephaloceles": "Thoát vị sọ não (cephalocele)",
    "structural_cerebral_atrophy": "Teo não",
    "structural_chiari_malformation": "Dị dạng Chiari",
    "structural_edema": "Phù não",
    "structural_encephalomalacia": "Nhuyễn não",
    "structural_mass_effect": "Hiệu ứng khối",
    "structural_midline_shift": "Lệch đường giữa",
    "surgical_catheter": "Ống thông/catheter sau can thiệp",
    "surgical_craniotomy": "Tình trạng sau mở sọ",
    "surgical_resection_cavity": "Hốc phẫu thuật cắt bỏ",
    "trauma_brain_contusion": "Dập não",
    "trauma_diffuse_axonal_injury": "Tổn thương sợi trục lan tỏa",
    "trauma_subdural_hematoma": "Tụ máu dưới màng cứng",
    "tumor_adult_brain_metastasis": "Di căn não ở người lớn",
    "tumor_adult_glioma": "U thần kinh đệm ở người lớn",
    "tumor_adult_high_grade_glioma": "U thần kinh đệm độ cao ở người lớn",
    "tumor_adult_low_grade_glioma": "U thần kinh đệm độ thấp ở người lớn",
    "tumor_adult_lymphoma": "Lymphoma não ở người lớn",
    "tumor_adult_meningioma": "U màng não ở người lớn",
    "tumor_adult_pineal_tumor": "U vùng tuyến tùng ở người lớn",
    "tumor_adult_schwannoma": "U bao dây thần kinh ở người lớn",
    "tumor_pediatric_brainstem_glioma": "U thần kinh đệm thân não ở trẻ em",
    "tumor_pediatric_ependymoma": "Ependymoma ở trẻ em",
    "tumor_pediatric_germ_cell_tumor": "U tế bào mầm ở trẻ em",
    "tumor_pediatric_medulloblastoma": "U nguyên bào tủy ở trẻ em",
    "tumor_pediatric_pilocytic_astrocytoma": "U sao bào lông ở trẻ em",
    "vascular_hemorrhagic_intracranial_hemorrhage": "Xuất huyết nội sọ",
    "vascular_hemorrhagic_intraventricular_hemorrhage": "Xuất huyết não thất",
    "vascular_hemorrhagic_subarachnoid_hemorrahage": "Xuất huyết dưới nhện",
    "vascular_ischemic_lacunar_stroke": "Nhồi máu ổ khuyết",
    "vascular_ischemic_large_vessel_stroke": "Nhồi máu mạch lớn",
    "vascular_ischemic_moyamoya": "Bệnh Moyamoya",
    "vascular_ischemic_small_vessel_disease": "Bệnh mạch máu nhỏ não",
    "vascular_malformation_arteriovenous_malformation": "Dị dạng động-tĩnh mạch",
    "vascular_malformation_cavernoma": "Dị dạng mạch hang (cavernoma)",
    "vascular_malformation_cerebral_aneurysm": "Phình động mạch não",
    "ventricular_intracranial_hypotension": "Hạ áp lực nội sọ",
    "ventricular_ventriculomegaly": "Giãn não thất",
}

REFERRAL_LABELS_VI = {
    "ns-pediatric": "Ngoại thần kinh nhi",
    "ns-skull base": "Ngoại thần kinh nền sọ",
    "ns-general": "Ngoại thần kinh tổng quát",
    "ns-trauma": "Ngoại thần kinh chấn thương",
    "ns-tumor": "Ngoại thần kinh u não",
    "ns-vascular": "Ngoại thần kinh mạch máu",
    "nl-pediatric": "Thần kinh nhi",
    "nl-epilepsy": "Thần kinh động kinh",
    "nl-neurocritical": "Hồi sức thần kinh",
    "nl-neuroimmunology": "Thần kinh miễn dịch",
    "nl-neurocognitive": "Thần kinh nhận thức",
    "nl-oncology": "Ung bướu thần kinh",
    "nl-trauma": "Thần kinh — chấn thương",
    "nl-stroke": "Thần kinh đột quỵ",
    "nl-general": "Thần kinh tổng quát",
}

DIAGNOSIS_GROUP_LABELS_VI = {
    "vascular_ischemic": "Mạch máu — thiếu máu não",
    "vascular_hemorrhagic": "Mạch máu — xuất huyết",
    "vascular_malformation": "Mạch máu — dị dạng mạch",
    "tumor": "Khối u",
    "infectious_inflammatory": "Nhiễm trùng / viêm",
    "trauma": "Chấn thương",
    "structural": "Cấu trúc / biến đổi nhu mô",
    "sellar": "Vùng yên — trên yên",
    "cyst": "Nang",
    "developmental": "Bất thường phát triển",
    "ventricular": "Não thất / dịch não tủy",
    "surgical": "Sau phẫu thuật / can thiệp",
    "spine": "Tủy sống",
    "other": "Khác",
}

DIAGNOSIS_GROUP_ORDER = [
    "vascular_ischemic",
    "vascular_hemorrhagic",
    "vascular_malformation",
    "tumor",
    "infectious_inflammatory",
    "trauma",
    "structural",
    "sellar",
    "cyst",
    "developmental",
    "ventricular",
    "surgical",
    "spine",
    "other",
]


def diagnosis_group(code: str) -> str:
    normalized = normalize_code(code)
    if normalized.startswith("vascular_ischemic_"):
        return "vascular_ischemic"
    if normalized.startswith("vascular_hemorrhagic_"):
        return "vascular_hemorrhagic"
    if normalized.startswith("vascular_malformation_"):
        return "vascular_malformation"
    if normalized.startswith("tumor_"):
        return "tumor"
    if normalized.startswith("infectious_") or normalized.startswith("inflammatory_"):
        return "infectious_inflammatory"
    if normalized.startswith("trauma_"):
        return "trauma"
    if normalized.startswith("structural_"):
        return "structural"
    if normalized.startswith("sellar_"):
        return "sellar"
    if normalized.startswith("cyst_"):
        return "cyst"
    if normalized.startswith("developmental_"):
        return "developmental"
    if normalized.startswith("ventricular_"):
        return "ventricular"
    if normalized.startswith("surgical_"):
        return "surgical"
    if normalized.startswith("spine_"):
        return "spine"
    return "other"


def grouped_diagnoses(rows):
    grouped = {key: [] for key in DIAGNOSIS_GROUP_ORDER}
    for name, score in rows:
        grouped[diagnosis_group(name)].append((name, score))
    return {
        key: grouped[key]
        for key in DIAGNOSIS_GROUP_ORDER
        if grouped[key]
    }



def scalar(value: Any) -> float:
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Prediction score must be finite")
        return number
    flat = value
    while isinstance(flat, list) and len(flat) == 1:
        flat = flat[0]
    if isinstance(flat, (int, float)):
        number = float(flat)
        if not math.isfinite(number):
            raise ValueError("Prediction score must be finite")
        return number
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
        if code in REFERRAL_LABELS_VI:
            return REFERRAL_LABELS_VI[code]
        if normalized in DIAGNOSIS_LABELS_VI:
            return DIAGNOSIS_LABELS_VI[normalized]
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
        grouped_positive = grouped_diagnoses(d_pos)
        grouped_near = grouped_diagnoses(d_near)

        priority_text = "Không có dữ liệu triage"
        if p_decision:
            p_name, _p_score = p_decision
            priority_text = PRIORITY_VI.get(
                normalize_code(p_name),
                label_for(p_name, language),
            )

        lines = [
            f"# PRIMA — Tóm tắt MRI não — {study_id}",
            "",
            "> **Mục đích:** hỗ trợ bác sĩ CĐHA rà soát study ở mức phân loại toàn bộ ca. "
            "PRIMA không định khu tổn thương và không thay thế việc đọc ảnh.",
            "",
            "## Đọc nhanh",
            "",
            f"- **Triage của mô hình:** {priority_text}",
            f"- **Nhãn chẩn đoán vượt ngưỡng:** {len(d_pos)}",
            f"- **Nhãn sát ngưỡng cần lưu ý:** {len(d_near)}",
            f"- **Gợi ý chuyên khoa vượt ngưỡng:** {len(r_pos)}",
            "",
        ]

        lines += [
            "## Các nhãn PRIMA vượt ngưỡng",
            "",
            "> Đây là các **study-level flags** của mô hình. Một ca có thể có nhiều nhãn cùng vượt ngưỡng; "
            "không nên diễn giải chúng như một kết luận CĐHA hoàn chỉnh.",
            "",
        ]
        if grouped_positive:
            for group_key, rows in grouped_positive.items():
                lines += [f"### {DIAGNOSIS_GROUP_LABELS_VI[group_key]}", ""]
                for name, _score in rows:
                    lines.append(f"- **{label_for(name, language)}**")
                lines.append("")
        else:
            lines += [
                "_Không có nhãn PRIMA nào vượt ngưỡng 0._",
                "",
                "**Lưu ý:** điều này không đồng nghĩa MRI bình thường.",
                "",
            ]

        if grouped_near:
            lines += [
                "## Các nhãn sát ngưỡng — nên nhìn lại ảnh nếu phù hợp lâm sàng",
                "",
            ]
            for group_key, rows in grouped_near.items():
                lines += [f"### {DIAGNOSIS_GROUP_LABELS_VI[group_key]}", ""]
                for name, _score in rows:
                    lines.append(f"- {label_for(name, language)}")
                lines.append("")

        lines += ["## Gợi ý hội chẩn / chuyên khoa", ""]
        if r_pos:
            for name, _score in r_pos:
                lines.append(f"- **{label_for(name, language)}**")
        else:
            lines.append("_Không có referral head nào vượt ngưỡng._")

        if r_near:
            lines += ["", "### Referral sát ngưỡng", ""]
            for name, _score in r_near:
                lines.append(f"- {label_for(name, language)}")

        lines += [
            "",
            "## Giới hạn cần nhớ khi đọc kết quả",
            "",
            "- PRIMA chỉ trả về **nhãn ở mức toàn study**; không cho biết vị trí tổn thương.",
            "- Không cung cấp kích thước, số lượng, laterality, đặc điểm T1/T2/FLAIR/DWI/ADC/SWI hay kiểu ngấm thuốc.",
            "- Score của diagnosis/referral là margin so với ngưỡng mô hình, **không phải xác suất bệnh**.",
            "- Không có nhãn vượt ngưỡng **không loại trừ** bất thường trên MRI.",
            "- Kết quả nên dùng như một danh sách gợi ý để rà lại ảnh, không dùng thay báo cáo CĐHA.",
            "",
        ]

        if show_all_scores:
            lines += ["## Chi tiết mô hình", ""]
            lines += ["### Diagnosis", ""]
            for name, score in d_pos + d_near + d_neg:
                lines.append(
                    f"- {label_for(name, language)} — margin `{fmt_margin(score)}`  "
                    f"<small>({name})</small>"
                )
            lines += ["", "### Referral", ""]
            for name, score in r_pos + r_near + r_neg:
                lines.append(
                    f"- {label_for(name, language)} — margin `{fmt_margin(score)}`  "
                    f"<small>({name})</small>"
                )
            lines += ["", "### Priority", ""]
            for name, score in sorted_scores(priority):
                lines.append(f"- {name}: `{fmt_margin(score)}`")
            lines.append("")

        lines += ["## Thông tin kỹ thuật", ""]
        if tech:
            lines.append(f"- Series xử lý: **{tech['series_count']}**")
            lines.append(f"- Series lỗi/bị bỏ: **{tech['skipped_series_count']}**")
            if tech.get("total_seconds") is not None:
                lines.append(f"- Tổng thời gian: **{tech['total_seconds']:.1f} giây**")
            if tech.get("gpu_name"):
                lines.append(f"- GPU: {tech['gpu_name']}")
        else:
            lines.append("_Không có runtime metrics._")

        lines += [
            "",
            "---",
            "",
            "Chi tiết score/mã task được ẩn khỏi phần đọc nhanh. "
            "Raw output vẫn được giữ riêng trong file `*_predictions.json` để máy xử lý.",
        ]
        return "\n".join(lines) + "\n"

    lines = [
        f"# PRIMA MRI Brain — {study_id}",
        "",
        "> **Note:** Research decision-support output only. Diagnosis/referral values are "
        "threshold margins; priority uses relative class scores and argmax. None are calibrated "
        "disease probabilities, and diagnosis tasks are independent.",
        "",
        "## Priority",
        "",
    ]
    if p_decision:
        p_name, p_score = p_decision
        lines.append(
            f"**{PRIORITY_EN.get(normalize_code(p_name), label_for(p_name, language))}** "
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


def markdown_to_html(markdown_text: str, *, study_id: str, evidence_html: str = "") -> str:
    body: List[str] = []
    in_list = False
    evidence_inserted = False

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
        if line.startswith("## Thông tin kỹ thuật") or line.startswith("## Technical information"):
            close_list()
            if evidence_html and not evidence_inserted:
                body.append(evidence_html)
                evidence_inserted = True
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line == "---":
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

    if evidence_html and not evidence_inserted:
        body.append(evidence_html)

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
.evidence-panel {{ margin-top:28px; }}
.evidence-disclaimer {{ margin-bottom:16px; font-size:14px; }}
.evidence-item {{ margin-bottom:24px; padding-bottom:16px; border-bottom:1px dashed #e0e0e0; }}
.evidence-item:last-child {{ border-bottom:none; }}
.evidence-meta {{ font-size:14px; color:#555; margin:4px 0 10px 0; }}
.evidence-grid {{ display:flex; flex-wrap:wrap; gap:12px; margin:8px 0; }}
.evidence-card {{ display:flex; flex-direction:column; align-items:center; background:#f8f9fa; padding:6px; border-radius:8px; border:1px solid #e2e4e8; }}
.evidence-card img {{ width:140px; height:140px; object-fit:cover; border-radius:4px; display:block; }}
.evidence-card span {{ font-size:11px; margin-top:5px; color:#555; font-weight:500; }}
@media (prefers-color-scheme: dark) {{
  body {{ background:#15171a; color:#e8eaed; }}
  main {{ background:#202124; box-shadow:none; }}
  h2 {{ border-color:#3c4043; }}
  .notice {{ background:#2b2d31; }}
  code {{ background:#303134; }}
  .evidence-item {{ border-color:#3c4043; }}
  .evidence-meta {{ color:#aaa; }}
  .evidence-card {{ background:#2a2b2e; border-color:#3c4043; }}
  .evidence-card span {{ color:#aaa; }}
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
    study_dir: Optional[Path] = None,
    metrics: Optional[Dict[str, Any]] = None,
    language: str = "vi",
    near_margin: float = 0.25,
    show_all_scores: bool = False,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if study_dir is None:
        candidate = (metrics or {}).get("config", {}).get("study_dir")
        if candidate and candidate != "<redacted>" and Path(candidate).exists():
            study_dir = Path(candidate)

    if study_dir is None:
        local_cfg = Path.home() / ".config" / "prima-antigravity" / "local.json"
        if local_cfg.exists():
            try:
                with open(local_cfg, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                r_root = cfg.get("PRIMA_RUNTIME_ROOT") or cfg.get("runtime_root")
                if r_root:
                    staged_path = Path(r_root) / "cases" / study_id
                    if staged_path.exists():
                        study_dir = staged_path
            except Exception:
                pass

    if study_dir is None:
        for cand in [Path("cases") / study_id, output_dir.parent / "cases" / study_id]:
            if cand.exists():
                study_dir = cand
                break

    evidence_html = ""
    if study_dir and generate_study_evidence:
        d_pos, _, _ = classify_threshold_group(predictions.get("diagnosis", {}), near_margin)
        positive_codes = [name for name, _ in d_pos]
        if positive_codes:
            try:
                evidence_entries = generate_study_evidence(
                    study_dir=study_dir,
                    output_dir=output_dir,
                    positive_diagnoses=positive_codes,
                    label_resolver=label_for,
                    language=language,
                )
                if build_evidence_html:
                    evidence_html = build_evidence_html(evidence_entries, language=language)
            except Exception:
                evidence_html = ""

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
    html_path.write_text(markdown_to_html(markdown, study_id=study_id, evidence_html=evidence_html), encoding="utf-8")
    return md_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PRIMA human-readable report")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--study-id")
    parser.add_argument("--study-dir", type=Path)
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
        study_dir=args.study_dir,
        metrics=metrics,
        language=args.language,
        near_margin=args.near_margin,
        show_all_scores=args.show_all_scores,
    )
    print(md_path)
    print(html_path)


if __name__ == "__main__":
    main()
