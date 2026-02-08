from __future__ import annotations

from typing import Any, Dict, List
from PIL import Image
import io

from ultralytics import YOLO

# -----------------------------
# Models + class config
# -----------------------------

# Your trained PPE model
_yolo_model = YOLO("runs/mining_ppe/mining_ppe_yolov8n4/weights/best.pt")

# Generic person detector (COCO-pretrained). Downloads once if not present.
_person_model = YOLO("yolov8n.pt")

# Classes from your mining PPE dataset
KEEP_CLASSES = {
    "boots",
    "helmet",
    "no-boots",
    "no-helmet",
    "no-vest",
    "vest",
    # ignore "undefined" for now
}

PPE_NON_COMPLIANT = {"no-boots", "no-helmet", "no-vest"}
PPE_COMPLIANT = {"boots", "helmet", "vest"}


def run_yolo_on_image_bytes(image_bytes: bytes, conf: float = 0.25) -> List[Dict[str, Any]]:
    """
    Two-stage detection:
      1) detect persons with COCO model (class 0)
      2) crop each person and run PPE model on the crop
    Returns detections in ORIGINAL image coordinates:
      [{label, conf, box_xyxy: [x1,y1,x2,y2]}]
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 1) person detection (only class 0 = person)
    person_results = _person_model.predict(img, conf=0.35, classes=[0])
    pr = person_results[0]

    person_boxes: List[tuple[int, int, int, int]] = []
    for b in pr.boxes:
        x1, y1, x2, y2 = b.xyxy.tolist()[0]

        # Filter tiny people to avoid crowds far away triggering PPE labels
        if (x2 - x1) < 80 or (y2 - y1) < 140:
            continue

        person_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    if not person_boxes:
        print("PPE detections: [] (no person boxes after filtering)")
        return []

    detections: List[Dict[str, Any]] = []

    # 2) PPE detection on each person crop
    for (px1, py1, px2, py2) in person_boxes:
        crop = img.crop((px1, py1, px2, py2))

        ppe_results = _yolo_model.predict(crop, conf=conf)
        r = ppe_results[0]
        names = r.names  # class_id -> name

        for bb in r.boxes:
            cls_id = int(bb.cls.item())
            label = names.get(cls_id, str(cls_id))

            if label not in KEEP_CLASSES:
                continue

            cx1, cy1, cx2, cy2 = bb.xyxy.tolist()[0]

            # Shift crop coords back into full-image coords
            detections.append(
                {
                    "label": label,
                    "conf": float(bb.conf.item()),
                    "box_xyxy": [
                        float(cx1 + px1),
                        float(cy1 + py1),
                        float(cx2 + px1),
                        float(cy2 + py1),
                    ],
                }
            )

    detections.sort(key=lambda d: d["conf"], reverse=True)
    print("PPE detections:", detections)
    return detections


def build_hazard_summary(detections: List[Dict[str, Any]]) -> str:
    """
    Convert detections into a short PPE compliance summary for RAG prompting.
    """
    counts: Dict[str, int] = {}
    for d in detections:
        label = d["label"]
        counts[label] = counts.get(label, 0) + 1

    if not counts:
        return (
            "Detected: helmet=0, no-helmet=0, vest=0, no-vest=0, "
            "boots=0, no-boots=0. risk_level=unknown."
        )

    helmets = counts.get("helmet", 0)
    no_helmets = counts.get("no-helmet", 0)
    vests = counts.get("vest", 0)
    no_vests = counts.get("no-vest", 0)
    boots = counts.get("boots", 0)
    no_boots = counts.get("no-boots", 0)

    non_compliant_total = no_helmets + no_vests + no_boots
    risk_level = "elevated" if non_compliant_total > 0 else "unknown"

    return (
        f"Detected: helmet={helmets}, no-helmet={no_helmets}, "
        f"vest={vests}, no-vest={no_vests}, boots={boots}, no-boots={no_boots}. "
        f"risk_level={risk_level}."
    )


def build_rag_question_from_hazards(hazard_summary: str) -> str:
    """
    Create a query that your RAG system can answer with grounded WHS controls.
    """
    return (
        "You are given a mining site photo PPE analysis.\n"
        f"{hazard_summary}\n\n"
        "Treat 'helmet' as safety helmet, 'vest' as high-visibility vest, and 'boots' as safety boots.\n"
        "Provide a short WHS checklist (3–7 bullets) of practical controls to manage PPE compliance "
        "for mining operations, including supervision, training, access control, and stop-work/escalation "
        "when critical PPE is missing. Cite sources using [1], [2] from the provided context."
    )
