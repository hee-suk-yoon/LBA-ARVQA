import os
import json
import base64
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import cv2
import ray

# Optional deps the original script imported (kept for parity)
import torch  # noqa: F401
from tqdm import tqdm  # noqa: F401
from transformers import AutoTokenizer  # noqa: F401
from datasets import load_dataset, Dataset  # noqa: F401

from openai import OpenAI

from google import genai
from google.genai import types

from google.genai.types import HttpOptions, Part


# ---------------------------- #
# Args & run name helpers
# ---------------------------- #
def set_runname(args):
    from datetime import date, datetime, timezone, timedelta
    KST = timezone(timedelta(hours=9))
    args.date = str(date.today())
    time_record = str(datetime.now(KST).time())[:8]
    args.run_name = args.date + "_" + time_record
    return args


def get_arguments():
    parser = argparse.ArgumentParser(description="Generate ambiguous video QA dataset with GPT API")

    parser.add_argument("--seed", type=int, default=4096)
    parser.add_argument("--data_root_dir", type=str, default=None)
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--debugging", action="store_true")

    # Prefer env var fallback instead of hardcoding a key
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    # Prefer env var fallback instead of hardcoding a key
    parser.add_argument(
        "--gemini-api_key",
        type=str,
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument("--gemini_version", type=str, default="gemini-2.5-flash", help="Gemini model version")

    parser.add_argument("--multi_process", action="store_true")

    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--frame_home_path", type=str, default=None)

    parser.add_argument("--project_id", type=str, default="lba-ambiguousqa", help="GCP Project ID for authentication")
    parser.add_argument("--location", type=str, default="us-central1", help="GCP Location for the Vertex AI endpoint")
    args = parser.parse_args()
    return args


# ---------------------------- #
# Frame extraction / loading
# ---------------------------- #
def get_video_frames(video_path, num_frames=64):
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    if len(base64Frames) > num_frames:
        sample_idx = np.linspace(0, len(base64Frames) - 1, num_frames)
        sampled_frames = [base64Frames[int(idx)] for idx in sample_idx]
    else:
        sampled_frames = base64Frames
    video.release()
    return sampled_frames


def load_preprocessed_video(json_path: str) -> List[str]:
    """
    Load the saved JSON back into Python as the exact list[str] returned by get_video_frames.
    """
    p = Path(json_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or (len(data) > 0 and not isinstance(data[0], str)):
        raise ValueError(f"Unexpected data format in {json_path} (expected list[str]).")
    return data


# ----------------------------
# GPT call
# ----------------------------
def get_response(args, video_frames):
    
    # Programmatically get an access token
    # client = OpenAI(api_key=args.api_key)
    client = genai.Client(api_key=args.gemini_api_key)
    
    frames = [types.Part.from_bytes(data=byte_, mime_type='image/jpeg') for byte_ in video_frames]
    text = [types.Part(
                text=("Inspect the video frames and create ONE ambiguous, video-grounded question "
                    "whose answer is present in the video but becomes CLEAR only after a brief user clarification. "
                    "Use only what is visible. "
                    "Ambiguous question must allow ≥2 interpretations; don’t pre-specify subject/time/space\n"
                    "Ambiguity types: referential|spatial|temporal|action|\n"
                    "- referential: unclear which entity is meant\n"
                    "- spatial: unclear location/frame of reference\n"
                    "- temporal: unclear time segment\n"
                    "- action: unclear action label for motions\n"
                    "Output ONLY a JSON object with keys: ambiguous_question, ambiguity_type, "
                    "clarifying_question, clarifying_answer_user (a short phrase), final_answer (answer once clarified); OR the exact string None if no ambiguous case is not present. "
                    "No extra text.")
            ),
            types.Part(
                text="Frames below. Follow the given rules and return only the JSON object or None.",
                ),]
    
    contents = types.Content(
        parts= text + frames
    )
    completion = client.models.generate_content(
        model=args.gemini_version,
        contents=contents,
    )
    response_message = completion.text.strip()
    return response_message


# ----------------------------
# Ray worker (safe: never lets third-party exceptions escape)
# ----------------------------
@ray.remote
def get_process_video_and_response(args, video_file):
    try:
        # If reading from mp4s, use get_video_frames; here we read preprocessed JSON.
        video_frames = load_preprocessed_video(video_file)
        response = get_response(args, video_frames)
        return {"ok": True, "video_file": video_file, "response": response}
    except Exception as e:
        return {
            "ok": False,
            "video_file": video_file,
            "error_type": type(e).__name__,
            "error": str(e),
            "trace": traceback.format_exc(),
        }


# ----------------------------
# Saving helpers (real-time)
# ----------------------------
def infer_dataset_name(video_json_path: str) -> str:
    """
    Try to infer dataset name from a path like:
    /.../MOMA-LRG/videos/raw/preprocessed_frames/clip123.json
    """
    p = Path(video_json_path)
    parts = p.parts
    if "preprocessed_frames" in parts:
        i = parts.index("preprocessed_frames") - 1
        # Walk back over generic directories like 'videos' or 'raw'
        while i >= 0 and parts[i] in ("videos", "raw"):
            i -= 1
        if i >= 0:
            return parts[i]
    # Fallback markers if structure differs
    for marker in ("MOMA-LRG", "Charades_v1", "ActionAtlas"):
        if marker in parts:
            return marker
    return "unknown"


def save_one_result(item: Dict[str, Any]) -> Path:
    """
    Save the GPT response to CWD/<dataset>/<basename>.json
    - If the model returned a JSON object, save it as JSON.
    - If it returned the exact string None, save that raw text (to preserve the contract).
    """
    vf_path = Path(item["video_file"])  # e.g., .../preprocessed_frames/clip123.json
    ds_name = infer_dataset_name(str(vf_path))
    out_dir = Path.cwd() / "Gemini" /ds_name 
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / vf_path.name  # same filename as frames JSON

    txt = item["response"]
    start_index = txt.find('{')
    end_index = txt.rfind('}')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        json_string = txt[start_index : end_index + 1]
    try:
        obj = json.loads(txt)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=4)
    
    except Exception:
        # Not valid JSON (likely exact string None) -> write raw text
        try:
            obj = json.loads(json_string)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=4)
        except Exception:
            with out_path.open("w", encoding="utf-8") as f:
                f.write(txt if txt.endswith("\n") else txt + "\n")

    print(f"[SAVED] {out_path}")
    return out_path


def save_error(item: Dict[str, Any]) -> Path:
    """
    Save an error report alongside successful outputs for inspection.
    """
    vf_path = Path(item["video_file"])
    ds_name = infer_dataset_name(str(vf_path))
    out_dir = Path.cwd() / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)
    err_path = out_dir / (vf_path.stem + ".error.txt")
    with err_path.open("w", encoding="utf-8") as f:
        f.write(f"{item.get('error_type')}: {item.get('error')}\n\n{item.get('trace','')}")
    print(f"[ERROR] {vf_path.name}: {item.get('error_type')} — see {err_path}")
    return err_path


# ----------------------------
# Main
# ----------------------------
def main():
    args = get_arguments()

    if args.debugging:
        ray.init(num_cpus=1)
    else:
        ray.init()

    output_save_path = os.path.join(args.root_dir, "dataset")
    if args.debugging:
        output_save_path = os.path.join(args.root_dir, "debugging_dataset_gemini")
    os.makedirs(output_save_path, exist_ok=True)

    # Preprocessed frames roots
    video_preprocessed_path_list = [
        "/data2/esyoon_hdd/MOMA-LRG/videos/raw/preprocessed_frames",
        "/data2/esyoon_hdd/Charades_v1/preprocessed_frames",
        "/data2/esyoon_hdd/ActionAtlas/videos/preprocessed_frames",
    ]

    # Collect all frames JSON files (skip missing dirs safely)
    video_files = []
    for video_path in video_preprocessed_path_list:
        if not os.path.isdir(video_path):
            print(f"[WARN] Missing directory: {video_path}")
            continue
        for f in os.listdir(video_path):
            if f.endswith(".json"):
                video_files.append(os.path.join(video_path, f))

    video_files = video_files[2:10] if args.debugging else video_files
    if not video_files:
        print("[INFO] No preprocessed frame JSONs found. Exiting.")
        return

    # Launch all tasks
    obj_refs = [get_process_video_and_response.remote(args, vf) for vf in video_files]

    # Save results as each one finishes (real-time)
    pending = set(obj_refs)
    while pending:
        ready, pending = ray.wait(list(pending), num_returns=1, timeout=None)
        for ref in ready:
            item = ray.get(ref)
            if item.get("ok"):
                save_one_result(item)
            else:
                save_error(item)

    print("All done.")


if __name__ == "__main__":
    main()
