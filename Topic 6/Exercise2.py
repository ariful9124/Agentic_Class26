# """
# exercise2_person_entry_exit.py
# Exercise 2: Scan video frames with Ollama LLaVA to detect when a person enters/exits.

# What this does
# - Loads frames from a folder (sorted by filename in *natural numeric* order)
# - For each frame, asks LLaVA: "Is there a person in the scene?"
# - Aggregates detections into contiguous "person present" segments
# - Prints entry/exit times (seconds) and also saves a CSV summary

# Assumptions
# - Frames are image files (jpg/png/webp) in a single folder
# - Filenames represent correct temporal order (supports both zero-padded and non-padded)
# - You know FPS (frames per second). Set FPS below.

# Deps
#   pip install pillow ollama

# Run
#   python exercise2_person_entry_exit.py --frames_dir /path/to/frames --fps 30

# Notes
# - This is intentionally simple (sequential). If you want speed, I can add batching/parallelism.
# - LLaVA can be noisy. We include smoothing: require K consecutive positives to start,
#   and K consecutive negatives to end.
# - This version hardens JSON parsing and boolean coercion to avoid common LLaVA output issues.
# """

# from __future__ import annotations

# import os
# import io
# import csv
# import json
# import base64
# import argparse
# import re
# from dataclasses import dataclass
# from typing import List, Dict, Any, Optional, Tuple

# from PIL import Image
# import ollama  # pip install ollama


# # -----------------------------
# # Configuration
# # -----------------------------
# OLLAMA_MODEL = "llava"          # e.g., "llava"
# MAX_IMAGE_SIDE = 1080            # downscale for speed (set higher if person is small)
# JPEG_QUALITY = 90               # jpeg encode quality (set higher if person is small)
# TEMPERATURE = 0.0               # deterministic-ish

# # Prompt: keep it STRICT so parsing is easy.
# # We force JSON output with a boolean.
# DETECT_PROMPT = (
#     "You are a strict image classifier.\n"
#     "Output ONLY valid JSON.\n"
#     "No explanations. No markdown.\n"
#     "Is there at least one visible human person in this image?\n"
#     '{"person": true/false, "confidence": 0.0-1.0}'
# )


# # Smoothing / debouncing:
# # - Need START_K consecutive positives to declare "entered"
# # - Need END_K consecutive negatives to declare "exited"
# START_K = 1
# END_K = 1

# # Optional: skip frames for speed (e.g., check every 2nd frame). Keep 1 for full scan.
# FRAME_STRIDE = 1


# # -----------------------------
# # Helpers
# # -----------------------------
# def downscale_image(img: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
#     """Downscale so max(width,height) <= max_side, preserving aspect ratio."""
#     img = img.convert("RGB")
#     w, h = img.size
#     m = max(w, h)
#     if m <= max_side:
#         return img
#     scale = max_side / float(m)
#     new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
#     return img.resize(new_size, Image.BICUBIC)


# def image_to_b64_jpeg(img: Image.Image, quality: int = JPEG_QUALITY) -> str:
#     """Encode PIL image as base64 JPEG (for Ollama vision models)."""
#     buf = io.BytesIO()
#     img.save(buf, format="JPEG", quality=quality, optimize=True)
#     return base64.b64encode(buf.getvalue()).decode("utf-8")


# def natural_key(s: str):
#     """
#     Natural sort key: "frame_2.jpg" < "frame_10.jpg".
#     Works for both padded and non-padded filenames.
#     """
#     return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# def list_frame_paths(frames_dir: str) -> List[str]:
#     exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
#     files = [
#         os.path.join(frames_dir, f)
#         for f in os.listdir(frames_dir)
#         if os.path.splitext(f.lower())[1] in exts
#     ]
#     files.sort(key=lambda p: natural_key(os.path.basename(p)))
#     return files


# def safe_parse_json(s: str) -> Optional[Dict[str, Any]]:
#     """
#     Try to parse a JSON object from model output.
#     - If the model includes extra text, we try to extract the first {...} block.
#     """
#     s = (s or "").strip()
#     if not s:
#         return None

#     # direct parse
#     try:
#         obj = json.loads(s)
#         return obj if isinstance(obj, dict) else None
#     except Exception:
#         pass

#     # extract first {...}
#     start = s.find("{")
#     end = s.rfind("}")
#     if start != -1 and end != -1 and end > start:
#         chunk = s[start : end + 1]
#         try:
#             obj = json.loads(chunk)
#             return obj if isinstance(obj, dict) else None
#         except Exception:
#             return None
#     return None


# def coerce_bool(v: Any) -> bool:
#     """
#     Robust boolean coercion:
#     - True/False stays as is
#     - Numeric: 0 -> False, nonzero -> True
#     - Strings: "true"/"false"/"yes"/"no"/"1"/"0" supported
#     This avoids the bug where bool("false") == True.
#     """
#     if isinstance(v, bool):
#         return v
#     if isinstance(v, (int, float)):
#         return v != 0
#     if isinstance(v, str):
#         s = v.strip().lower()
#         if s in {"true", "t", "yes", "y", "1"}:
#             return True
#         if s in {"false", "f", "no", "n", "0"}:
#             return False
#     return False


# def llava_person_present(img_b64: str) -> Tuple[bool, float, str, bool]:
#     """
#     Ask LLaVA if a person is present.
#     Returns: (person_bool, confidence_float, raw_text, parsed_ok)
#     """
#     messages = [
#         {
#             "role": "user",
#             "content": (
#                 DETECT_PROMPT +
#                 "\nRemember: Only return a single JSON object. Nothing else."
#             ),
#             "images": [img_b64],
#         }
#     ]


#     resp = ollama.chat(
#         model=OLLAMA_MODEL,
#         messages=messages,
#         options={"temperature": TEMPERATURE},
#     )

#     text = resp["message"]["content"]
#     data = safe_parse_json(text)
#     parsed_ok = data is not None
#     data = data or {}

#     person = coerce_bool(data.get("person", False))
#     conf = data.get("confidence", 0.0)

#     try:
#         conf = float(conf)
#     except Exception:
#         conf = 0.0

#     # clamp
#     conf = max(0.0, min(1.0, conf))
#     return person, conf, text, parsed_ok


# @dataclass
# class Segment:
#     start_frame: int
#     end_frame: int  # inclusive
#     start_time_s: float
#     end_time_s: float
#     mean_conf: float


# def frames_to_time(frame_idx: int, fps: float) -> float:
#     return frame_idx / float(fps)


# def detect_segments(
#     frame_paths: List[str],
#     fps: float,
#     stride: int = 1,
#     start_k: int = START_K,
#     end_k: int = END_K,
# ) -> Tuple[List[Segment], List[Dict[str, Any]]]:
#     """
#     Walk through frames, query person presence, apply debouncing,
#     and produce entry/exit segments + per-frame log.
#     """
#     segments: List[Segment] = []
#     log_rows: List[Dict[str, Any]] = []

#     in_segment = False
#     seg_start_frame: Optional[int] = None
#     seg_confs: List[float] = []

#     pos_run = 0
#     neg_run = 0

#     for idx in range(0, len(frame_paths), stride):
#         path = frame_paths[idx]

#         # load + encode (use context manager to close file handles)
#         with Image.open(path) as im:
#             img = downscale_image(im, MAX_IMAGE_SIDE)
#             img_b64 = image_to_b64_jpeg(img, JPEG_QUALITY)

#         person, conf, raw, parsed_ok = llava_person_present(img_b64)
#         print(f"Person: {person}, Confidence: {conf}, Parsed OK: {parsed_ok}")
#         log_rows.append(
#             {
#                 "frame_index": idx,
#                 "time_s": frames_to_time(idx, fps),
#                 "filename": os.path.basename(path),
#                 "person": int(person),
#                 "confidence": conf,
#                 "parsed_ok": int(parsed_ok),
#                 "raw": (raw or "").strip().replace("\n", " ")[:500],
#             }
#         )

#         if person:
#             pos_run += 1
#             neg_run = 0
#         else:
#             neg_run += 1
#             pos_run = 0

#         # Enter condition
#         if not in_segment and pos_run >= start_k:
#             # declare entry at the first frame of this positive run
#             entry_frame = idx - (start_k - 1) * stride
#             in_segment = True
#             seg_start_frame = entry_frame
#             seg_confs = []

#         # While in segment, collect confidence
#         if in_segment:
#             seg_confs.append(conf)

#         # Exit condition
#         if in_segment and neg_run >= end_k:
#             # declare exit at the last positive frame before the negative run began
#             exit_frame = idx - end_k * stride
#             # Guard in case stride causes overshoot
#             if seg_start_frame is None:
#                 seg_start_frame = 0
#             exit_frame = max(exit_frame, seg_start_frame)

#             start_time = frames_to_time(seg_start_frame, fps)
#             end_time = frames_to_time(exit_frame, fps)
#             mean_conf = sum(seg_confs) / max(1, len(seg_confs))

#             segments.append(
#                 Segment(
#                     start_frame=seg_start_frame,
#                     end_frame=exit_frame,
#                     start_time_s=start_time,
#                     end_time_s=end_time,
#                     mean_conf=mean_conf,
#                 )
#             )

#             in_segment = False
#             seg_start_frame = None
#             seg_confs = []
#             pos_run = 0
#             neg_run = 0

#     # If video ends while still "in segment", close it at last scanned frame
#     if in_segment and seg_start_frame is not None:
#         last_idx = ((len(frame_paths) - 1) // stride) * stride
#         start_time = frames_to_time(seg_start_frame, fps)
#         end_time = frames_to_time(last_idx, fps)
#         mean_conf = sum(seg_confs) / max(1, len(seg_confs))
#         segments.append(
#             Segment(
#                 start_frame=seg_start_frame,
#                 end_frame=last_idx,
#                 start_time_s=start_time,
#                 end_time_s=end_time,
#                 mean_conf=mean_conf,
#             )
#         )

#     return segments, log_rows


# def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=fieldnames)
#         w.writeheader()
#         for r in rows:
#             w.writerow(r)


# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--frames_dir", required=True, help="Folder containing extracted frames.")
#     ap.add_argument("--fps", type=float, required=True, help="Frames per second of the source video.")
#     ap.add_argument("--stride", type=int, default=FRAME_STRIDE, help="Process every Nth frame for speed.")
#     ap.add_argument("--start_k", type=int, default=START_K, help="Consecutive positives to start segment.")
#     ap.add_argument("--end_k", type=int, default=END_K, help="Consecutive negatives to end segment.")
#     ap.add_argument("--out_csv", default="frame_person_log.csv", help="Per-frame log CSV.")
#     ap.add_argument("--segments_csv", default="person_segments.csv", help="Entry/exit segments CSV.")
#     args = ap.parse_args()

#     frame_paths = list_frame_paths(args.frames_dir)
#     if not frame_paths:
#         raise SystemExit(f"No image frames found in: {args.frames_dir}")

#     print(f"Found {len(frame_paths)} frames in: {args.frames_dir}")
#     print(
#         f"Model: {OLLAMA_MODEL} | fps={args.fps} | stride={args.stride} "
#         f"| start_k={args.start_k} | end_k={args.end_k}"
#     )

#     # Optional: quick sanity check of ordering (first ~25)
#     # for i, p in enumerate(frame_paths[:25]):
#     #     print(i, os.path.basename(p))

#     segments, log_rows = detect_segments(
#         frame_paths=frame_paths,
#         fps=args.fps,
#         stride=args.stride,
#         start_k=args.start_k,
#         end_k=args.end_k,
#     )

#     # Save logs
#     save_csv(
#         args.out_csv,
#         log_rows,
#         fieldnames=["frame_index", "time_s", "filename", "person", "confidence", "parsed_ok", "raw"],
#     )

#     seg_rows = [
#         {
#             "start_frame": s.start_frame,
#             "end_frame": s.end_frame,
#             "start_time_s": round(s.start_time_s, 3),
#             "end_time_s": round(s.end_time_s, 3),
#             "duration_s": round(s.end_time_s - s.start_time_s, 3),
#             "mean_conf": round(s.mean_conf, 3),
#         }
#         for s in segments
#     ]
#     save_csv(
#         args.segments_csv,
#         seg_rows,
#         fieldnames=["start_frame", "end_frame", "start_time_s", "end_time_s", "duration_s", "mean_conf"],
#     )

#     # Print summary
#     if not segments:
#         print("\nNo person-presence segments detected.")
#         print(f"Per-frame log saved to: {args.out_csv}")
#         print("Tip: If parsed_ok is often 0, your model isn't returning strict JSON—check raw outputs.")
#         return

#     print("\nDetected person-presence segments (entry -> exit):")
#     for i, s in enumerate(segments, 1):
#         print(
#             f"  #{i}: {s.start_time_s:.2f}s (frame {s.start_frame})  ->  {s.end_time_s:.2f}s (frame {s.end_frame})"
#             f"   | duration {s.end_time_s - s.start_time_s:.2f}s | mean_conf {s.mean_conf:.2f}"
#         )

#     print(f"\nPer-frame log saved to: {args.out_csv}")
#     print(f"Segments saved to:     {args.segments_csv}")
#     print("\nTip: If detections flicker, increase --start_k and --end_k (e.g., 5).")
#     print("Tip: If small/far people are missed, raise MAX_IMAGE_SIDE / JPEG_QUALITY in the script.")


# if __name__ == "__main__":
#     main()
"""
exercise2_person_entry_exit.py
Exercise 2: Scan video frames with Ollama LLaVA to detect when a person enters/exits.

What this does
- Loads frames from a folder (sorted by filename in *natural numeric* order)
- For each frame, asks LLaVA: "Is there a person in the scene?"
- Aggregates detections into contiguous "person present" segments
- Prints entry/exit times (seconds) and also saves a CSV summary

Assumptions
- Frames are image files (jpg/png/webp) in a single folder
- Filenames represent correct temporal order (supports both zero-padded and non-padded)
- You know FPS (frames per second). Set FPS below.

Deps
  pip install pillow ollama

Run
  python exercise2_person_entry_exit.py --frames_dir /path/to/frames --fps 30

Notes
- This is intentionally simple (sequential). If you want speed, I can add batching/parallelism.
- LLaVA can be noisy. We include smoothing: require K consecutive positives to start,
  and K consecutive negatives to end.
- This version hardens JSON parsing and boolean coercion to avoid common LLaVA output issues.
"""

from __future__ import annotations

import os
import io
import csv
import json
import base64
import argparse
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image
import ollama  # pip install ollama


# -----------------------------
# Configuration
# -----------------------------
OLLAMA_MODEL = "llava"          # e.g., "llava"
MAX_IMAGE_SIDE = 1080            # downscale for speed (set higher if person is small)
JPEG_QUALITY = 90               # jpeg encode quality (set higher if person is small)
TEMPERATURE = 0.0               # deterministic-ish

# Prompt: keep it STRICT so parsing is easy.
# We force JSON output with a boolean.
DETECT_PROMPT = (
    "You are a strict image classifier.\n"
    "Output ONLY valid JSON.\n"
    "No explanations. No markdown.\n"
    "Is there at least one visible human person in this image?\n"
    '{"person": true/false, "confidence": 0.0-1.0}'
)


# Smoothing / debouncing:
# - Need START_K consecutive positives to declare "entered"
# - Need END_K consecutive negatives to declare "exited"
START_K = 1
END_K = 2

# Optional: skip frames for speed (e.g., check every 2nd frame). Keep 1 for full scan.
FRAME_STRIDE = 1


# -----------------------------
# Helpers
# -----------------------------
def downscale_image(img: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """Downscale so max(width,height) <= max_side, preserving aspect ratio."""
    img = img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.BICUBIC)


def image_to_b64_jpeg(img: Image.Image, quality: int = JPEG_QUALITY) -> str:
    """Encode PIL image as base64 JPEG (for Ollama vision models)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def natural_key(s: str):
    """
    Natural sort key: "frame_2.jpg" < "frame_10.jpg".
    Works for both padded and non-padded filenames.
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_frame_paths(frames_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    files = [
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if os.path.splitext(f.lower())[1] in exts
    ]
    files.sort(key=lambda p: natural_key(os.path.basename(p)))
    return files


def safe_parse_json(s: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse a JSON object from model output.
    - If the model includes extra text, we try to extract the first {...} block.
    """
    s = (s or "").strip()
    if not s:
        return None

    # direct parse
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # extract first {...}
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = s[start : end + 1]
        try:
            obj = json.loads(chunk)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def coerce_bool(v: Any) -> bool:
    """
    Robust boolean coercion:
    - True/False stays as is
    - Numeric: 0 -> False, nonzero -> True
    - Strings: "true"/"false"/"yes"/"no"/"1"/"0" supported
    This avoids the bug where bool("false") == True.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "t", "yes", "y", "1"}:
            return True
        if s in {"false", "f", "no", "n", "0"}:
            return False
    return False


def llava_person_present(image_path: str) -> Tuple[bool, float, str, bool]:
    """
    Ask LLaVA if a person is present.
    Returns: (person_bool, confidence_float, raw_text, parsed_ok)

    NOTE: Sends the ORIGINAL image file path to Ollama (no resizing / re-encode).
    """
    messages = [
        {
            "role": "user",
            "content": (
                DETECT_PROMPT +
                "\nRemember: Only return a single JSON object. Nothing else."
            ),
            "images": [image_path],
        }
    ]

    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={"temperature": TEMPERATURE},
    )

    text = resp["message"]["content"]
    data = safe_parse_json(text)
    parsed_ok = data is not None
    data = data or {}

    person = coerce_bool(data.get("person", False))
    conf = data.get("confidence", 0.0)

    try:
        conf = float(conf)
    except Exception:
        conf = 0.0

    # clamp
    conf = max(0.0, min(1.0, conf))
    return person, conf, text, parsed_ok


@dataclass
class Segment:
    start_frame: int
    end_frame: int  # inclusive
    start_time_s: float
    end_time_s: float
    mean_conf: float


def frames_to_time(frame_idx: int, fps: float) -> float:
    return frame_idx / float(fps)


def detect_segments(
    frame_paths: List[str],
    fps: float,
    stride: int = 1,
    start_k: int = START_K,
    end_k: int = END_K,
) -> Tuple[List[Segment], List[Dict[str, Any]]]:
    """
    Walk through frames, query person presence, apply debouncing,
    and produce entry/exit segments + per-frame log.
    """
    segments: List[Segment] = []
    log_rows: List[Dict[str, Any]] = []

    in_segment = False
    seg_start_frame: Optional[int] = None
    seg_confs: List[float] = []

    pos_run = 0
    neg_run = 0

    for idx in range(0, len(frame_paths), stride):
        path = frame_paths[idx]
        print(f'Processing frame {idx}')
        # ✅ Send ORIGINAL image (no resize / no base64)
        person, conf, raw, parsed_ok = llava_person_present(path)

        print(f"Person: {person}, Confidence: {conf}, Parsed OK: {parsed_ok}")
        log_rows.append(
            {
                "frame_index": idx,
                "time_s": frames_to_time(idx, fps),
                "filename": os.path.basename(path),
                "person": int(person),
                "confidence": conf,
                "parsed_ok": int(parsed_ok),
                "raw": (raw or "").strip().replace("\n", " ")[:500],
            }
        )

        if person:
            pos_run += 1
            neg_run = 0
        else:
            neg_run += 1
            pos_run = 0

        # Enter condition
        if not in_segment and pos_run >= start_k:
            # declare entry at the first frame of this positive run
            entry_frame = idx - (start_k - 1) * stride
            in_segment = True
            seg_start_frame = entry_frame
            seg_confs = []

        # While in segment, collect confidence
        if in_segment:
            seg_confs.append(conf)

        # Exit condition
        if in_segment and neg_run >= end_k:
            # declare exit at the last positive frame before the negative run began
            exit_frame = idx - end_k * stride
            # Guard in case stride causes overshoot
            if seg_start_frame is None:
                seg_start_frame = 0
            exit_frame = max(exit_frame, seg_start_frame)

            start_time = frames_to_time(seg_start_frame, fps)
            end_time = frames_to_time(exit_frame, fps)
            mean_conf = sum(seg_confs) / max(1, len(seg_confs))

            segments.append(
                Segment(
                    start_frame=seg_start_frame,
                    end_frame=exit_frame,
                    start_time_s=start_time,
                    end_time_s=end_time,
                    mean_conf=mean_conf,
                )
            )

            in_segment = False
            seg_start_frame = None
            seg_confs = []
            pos_run = 0
            neg_run = 0

    # If video ends while still "in segment", close it at last scanned frame
    if in_segment and seg_start_frame is not None:
        last_idx = ((len(frame_paths) - 1) // stride) * stride
        start_time = frames_to_time(seg_start_frame, fps)
        end_time = frames_to_time(last_idx, fps)
        mean_conf = sum(seg_confs) / max(1, len(seg_confs))
        segments.append(
            Segment(
                start_frame=seg_start_frame,
                end_frame=last_idx,
                start_time_s=start_time,
                end_time_s=end_time,
                mean_conf=mean_conf,
            )
        )

    return segments, log_rows


def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="Folder containing extracted frames.")
    ap.add_argument("--fps", type=float, required=True, help="Frames per second of the source video.")
    ap.add_argument("--stride", type=int, default=FRAME_STRIDE, help="Process every Nth frame for speed.")
    ap.add_argument("--start_k", type=int, default=START_K, help="Consecutive positives to start segment.")
    ap.add_argument("--end_k", type=int, default=END_K, help="Consecutive negatives to end segment.")
    ap.add_argument("--out_csv", default="frame_person_log.csv", help="Per-frame log CSV.")
    ap.add_argument("--segments_csv", default="person_segments.csv", help="Entry/exit segments CSV.")
    args = ap.parse_args()

    frame_paths = list_frame_paths(args.frames_dir)
    if not frame_paths:
        raise SystemExit(f"No image frames found in: {args.frames_dir}")

    print(f"Found {len(frame_paths)} frames in: {args.frames_dir}")
    print(
        f"Model: {OLLAMA_MODEL} | fps={args.fps} | stride={args.stride} "
        f"| start_k={args.start_k} | end_k={args.end_k}"
    )

    segments, log_rows = detect_segments(
        frame_paths=frame_paths,
        fps=args.fps,
        stride=args.stride,
        start_k=args.start_k,
        end_k=args.end_k,
    )

    # Save logs
    save_csv(
        args.out_csv,
        log_rows,
        fieldnames=["frame_index", "time_s", "filename", "person", "confidence", "parsed_ok", "raw"],
    )

    seg_rows = [
        {
            "start_frame": s.start_frame,
            "end_frame": s.end_frame,
            "start_time_s": round(s.start_time_s, 3),
            "end_time_s": round(s.end_time_s, 3),
            "duration_s": round(s.end_time_s - s.start_time_s, 3),
            "mean_conf": round(s.mean_conf, 3),
        }
        for s in segments
    ]
    save_csv(
        args.segments_csv,
        seg_rows,
        fieldnames=["start_frame", "end_frame", "start_time_s", "end_time_s", "duration_s", "mean_conf"],
    )

    # Print summary
    if not segments:
        print("\nNo person-presence segments detected.")
        print(f"Per-frame log saved to: {args.out_csv}")
        print("Tip: If parsed_ok is often 0, your model isn't returning strict JSON—check raw outputs.")
        return

    print("\nDetected person-presence segments (entry -> exit):")
    for i, s in enumerate(segments, 1):
        print(
            f"  #{i}: {s.start_time_s:.2f}s (frame {s.start_frame})  ->  {s.end_time_s:.2f}s (frame {s.end_frame})"
            f"   | duration {s.end_time_s - s.start_time_s:.2f}s | mean_conf {s.mean_conf:.2f}"
        )

    print(f"\nPer-frame log saved to: {args.out_csv}")
    print(f"Segments saved to:     {args.segments_csv}")
    print("\nTip: If detections flicker, increase --start_k and --end_k (e.g., 5).")
    print("Tip: If small/far people are missed, raise MAX_IMAGE_SIDE / JPEG_QUALITY in the script.")


if __name__ == "__main__":
    main()
