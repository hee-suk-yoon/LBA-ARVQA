import os
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional


# ===== Your preprocessing function (kept as-is, just with imports/types) =====
def get_video_frames(video_path: str, num_frames: int = 64) -> List[str]:
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    if len(base64Frames) > num_frames:
        sample_idx = np.linspace(0, len(base64Frames)-1, num_frames)
        sampled_frames = [base64Frames[int(idx)] for idx in sample_idx]
    else:
        sampled_frames = base64Frames
    video.release()
    return sampled_frames


# ===== Helpers =====
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_json(data, out_path: Path) -> None:
    """Write JSON atomically to avoid partial writes."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, separators=(",", ":"))
    os.replace(tmp_path, out_path)


def list_mp4s(source_dir: Path) -> List[Path]:
    """List .mp4 files (non-recursive by default). Set recursive=True if needed."""
    return sorted([p for p in source_dir.glob("*.mp4") if p.is_file()])


def select_unprocessed(
    source_dir: Path,
    save_dir: Path,
    n: int
) -> List[Tuple[Path, Path]]:
    """
    Return up to n tuples of (video_path, output_json_path) for videos
    that do not yet have a corresponding JSON in save_dir.
    Output filename is <mp4_basename>.json so it can be loaded exactly back as a list.
    """
    ensure_dir(save_dir)
    candidates = []
    for mp4 in list_mp4s(source_dir):
        # Output filename uses the same base name as the mp4, with .json extension
        out_json = save_dir / f"{mp4.stem}.json"
        if not out_json.exists():
            candidates.append((mp4, out_json))
        if len(candidates) >= n:
            break
    return candidates


def process_one(args: Tuple[Path, Path, int]) -> Tuple[str, str, Optional[str]]:
    """
    Worker function for multiprocessing.
    Returns (video_name, status, error_message_if_any)
    """
    video_path, out_json, num_frames = args
    try:
        frames_list = get_video_frames(str(video_path), num_frames=num_frames)
        # Save exactly what get_video_frames returns: a list[str].
        atomic_write_json(frames_list, out_json)
        return (video_path.name, "ok", None)
    except Exception as e:
        return (video_path.name, "error", str(e))


def preprocess_sources(
    video_path_list: List[str],
    n_per_source: int = 30,
    num_frames: int = 64,
    save_dir_name: str = "preprocessed_frames",  # created inside each source
    processes: Optional[int] = None,
) -> None:
    """
    For each source folder, create a saving directory and preprocess up to n_per_source
    videos that are NOT already saved there.
    """
    if processes is None:
        # Leave 1 core free by default, but ensure at least 1 worker.
        processes = max(1, (cpu_count() or 1) - 1)

    all_tasks: List[Tuple[Path, Path, int]] = []
    per_source_plan = []

    for src in video_path_list:
        source_dir = Path(src).expanduser().resolve()
        if not source_dir.exists():
            print(f"[WARN] Source not found: {source_dir}")
            continue

        save_dir = source_dir / save_dir_name
        to_process = select_unprocessed(source_dir, save_dir, n_per_source)
        per_source_plan.append((source_dir, save_dir, len(to_process)))
        for (video_path, out_json) in to_process:
            all_tasks.append((video_path, out_json, num_frames))

    # Plan summary
    print("\n=== Plan ===")
    for (source_dir, save_dir, count) in per_source_plan:
        print(f"Source: {source_dir}")
        print(f" Save to: {save_dir}")
        print(f" Selected: {count} to process\n")

    if not all_tasks:
        print("Nothing to do. All selected videos appear to be preprocessed already.")
        return

    print(f"Starting preprocessing with {processes} worker(s); total tasks: {len(all_tasks)}")
    with Pool(processes=processes) as pool:
        for video_name, status, err in pool.imap_unordered(process_one, all_tasks, chunksize=1):
            if status == "ok":
                print(f"[OK] {video_name}")
            else:
                print(f"[ERR] {video_name}: {err}")


# ===== Loader utility to verify outputs when needed =====
def load_preprocessed_video(json_path: str) -> List[str]:
    """
    Load the saved JSON back into Python as the exact list[str] returned by get_video_frames.
    """
    p = Path(json_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Optional: sanity check type/shape
    if not isinstance(data, list) or (len(data) > 0 and not isinstance(data[0], str)):
        raise ValueError(f"Unexpected data format in {json_path} (expected list[str]).")
    return data


if __name__ == "__main__":
    # Your source directories:
    video_path_list = [
        "VIDEO_FOLDER_PATHS_YOU_WANT_TO_PREPROCESS"
    ]

    # Configure as needed:
    N_PER_SOURCE = 1000       # <-- make this whatever you like
    NUM_FRAMES = 64         # frames sampled per video by get_video_frames
    SAVE_DIR_NAME = "preprocessed_frames"  # created inside each source directory

    preprocess_sources(
        video_path_list=video_path_list,
        n_per_source=N_PER_SOURCE,
        num_frames=NUM_FRAMES,
        save_dir_name=SAVE_DIR_NAME,
        processes=None,  # auto: cpu_count()-1
    )

    # Example of loading one output back (uncomment to use):
    # example_json = Path(video_path_list[0]) / SAVE_DIR_NAME / "example_video.json"
    # frames = load_preprocessed_video(str(example_json))
    # print(f"Loaded {len(frames)} frames from {example_json}")
