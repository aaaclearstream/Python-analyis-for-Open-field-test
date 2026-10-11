#!/usr/bin/env python3
# python3 "/Users/li/Desktop/20261007_OFT_analysis.py" --folder "/Users/li/Desktop/20261008_TS011_OFT_1M" --start 1 --end 11 --mm 2.7


from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

try:
    import cv2
    import numpy as np
except ImportError as exc:  # Give a useful message before any analysis starts.
    bundled_python = Path("/Users/li/Documents/Python scripts/.venv-dlc/bin/python")
    if bundled_python.is_file() and Path(sys.executable).resolve() != bundled_python.resolve():
        os.execv(str(bundled_python), [str(bundled_python), *sys.argv])
    raise SystemExit(
        "Missing packages. Install them with:\n"
        "  python3 -m pip install opencv-python numpy\n"
    ) from exc


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".m4v", ".mpg", ".mpeg"}
DEFAULT_LOCATION = Path("/Users/li/Desktop/Test")


@dataclass(frozen=True)
class Arena:
    center_x_px: float
    center_y_px: float
    radius_px: float
    method: str

    @property
    def center_radius_px(self) -> float:
        return self.radius_px / 2.0


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps


def video_info(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    info = VideoInfo(
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    if info.fps <= 0 or info.frame_count <= 0 or info.width <= 0 or info.height <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return info


def sample_background(path: Path, count: int = 9) -> np.ndarray:
    info = video_info(path)
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    for frame_number in np.linspace(0, info.frame_count - 1, count, dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not sample frames from {path}")
    return np.median(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)


def detect_arena(path: Path) -> tuple[Arena, np.ndarray]:
    """Detect the circular arena, including low-contrast white cylinders."""
    background = sample_background(path)
    gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2.0)
    height, width = gray.shape
    minimum_dimension = min(height, width)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=minimum_dimension * 0.5,
        param1=80,
        param2=45,
        minRadius=int(minimum_dimension * 0.25),
        maxRadius=int(minimum_dimension * 0.50),
    )
    if circles is not None:
        candidates = np.squeeze(circles, axis=0)
        if candidates.ndim == 1:
            candidates = candidates[None, :]
        image_center = np.array([width / 2.0, height / 2.0])
        scores = [c[2] - 0.20 * np.linalg.norm(c[:2] - image_center) for c in candidates]
        x, y, radius = candidates[int(np.argmax(scores))]
        return Arena(float(x), float(y), float(radius), "Hough circle"), background

    # White arena walls can have too little contrast for Hough detection. Search
    # for the circle whose circumference contains the most image-gradient energy.
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)

    def circumference_score(cx: float, cy: float, radius: float) -> float:
        xs = np.rint(cx + radius * np.cos(angles)).astype(int)
        ys = np.rint(cy + radius * np.sin(angles)).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        values = gradient[ys[valid], xs[valid]]
        if len(values) < 360:
            return -math.inf
        return float(np.mean(np.clip(values, 0, np.percentile(values, 90))))

    best_score = -math.inf
    best: tuple[float, float, float] | None = None
    for cx in np.linspace(width * 0.40, width * 0.60, 17):
        for cy in np.linspace(height * 0.40, height * 0.60, 17):
            for radius in np.linspace(minimum_dimension * 0.38, minimum_dimension * 0.49, 20):
                score = circumference_score(cx, cy, radius)
                if score > best_score:
                    best_score, best = score, (float(cx), float(cy), float(radius))
    if best is None:
        raise RuntimeError("Automatic arena detection failed")

    coarse_x, coarse_y, coarse_radius = best
    for cx in np.linspace(coarse_x - width * 0.015, coarse_x + width * 0.015, 13):
        for cy in np.linspace(coarse_y - height * 0.015, coarse_y + height * 0.015, 13):
            for radius in np.linspace(
                coarse_radius - minimum_dimension * 0.015,
                coarse_radius + minimum_dimension * 0.015,
                13,
            ):
                score = circumference_score(cx, cy, radius)
                if score > best_score:
                    best_score, best = score, (float(cx), float(cy), float(radius))
    return Arena(*best, method="radial gradient fit"), background


def find_blue_center(
    frame: np.ndarray,
    arena: Arena,
    previous: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Return the blue marker centroid in full-frame pixel coordinates."""
    margin = 20
    x1 = max(0, int(arena.center_x_px - arena.radius_px - margin))
    y1 = max(0, int(arena.center_y_px - arena.radius_px - margin))
    x2 = min(frame.shape[1], int(arena.center_x_px + arena.radius_px + margin) + 1)
    y2 = min(frame.shape[0], int(arena.center_y_px + arena.radius_px + margin) + 1)
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Broad blue range: tolerant of lighting changes while excluding the red tail.
    mask = cv2.inRange(hsv, np.array((90, 70, 35)), np.array((140, 255, 255)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

    candidates: list[tuple[float, float, float]] = []
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if not 20 <= area <= 12000:
            continue
        cx = float(centroids[index, 0] + x1)
        cy = float(centroids[index, 1] + y1)
        if math.hypot(cx - arena.center_x_px, cy - arena.center_y_px) > arena.radius_px * 1.08:
            continue
        candidates.append((area, cx, cy))
    if not candidates:
        return None
    if previous is None:
        _, cx, cy = max(candidates)
        return cx, cy

    # Prefer the expected marker size, with temporal proximity as a tie-breaker.
    def candidate_score(item: tuple[float, float, float]) -> float:
        area, cx, cy = item
        distance = math.hypot(cx - previous[0], cy - previous[1])
        return math.log1p(area) - 0.008 * distance

    _, cx, cy = max(candidates, key=candidate_score)
    return cx, cy


def interpolate_short_gaps(values: np.ndarray, maximum_gap: int) -> np.ndarray:
    result = values.copy()
    missing = ~np.isfinite(result)
    index = 0
    while index < len(result):
        if not missing[index]:
            index += 1
            continue
        start = index
        while index < len(result) and missing[index]:
            index += 1
        end = index
        if start > 0 and end < len(result) and end - start <= maximum_gap:
            result[start:end] = np.linspace(result[start - 1], result[end], end - start + 2)[1:-1]
    return result


def rolling_nanmedian(values: np.ndarray, window: int = 5) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    radius = window // 2
    for index in range(len(values)):
        portion = values[max(0, index - radius) : min(len(values), index + radius + 1)]
        if np.isfinite(portion).any():
            result[index] = float(np.nanmedian(portion))
    return result


def track_video(
    path: Path,
    arena: Arena,
    start_s: float,
    end_s: float,
    progress: Callable[[str], None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, VideoInfo, int, int]:
    info = video_info(path)
    if start_s < 0 or end_s <= start_s:
        raise ValueError("End time must be greater than start time")
    if start_s >= info.duration_s:
        raise ValueError(
            f"Start time ({start_s / 60:.2f} min) is beyond video duration "
            f"({info.duration_s / 60:.2f} min)"
        )
    clipped_end_s = min(end_s, info.duration_s)
    start_frame = max(0, int(round(start_s * info.fps)))
    end_frame = min(info.frame_count, int(round(clipped_end_s * info.fps)))
    frame_total = end_frame - start_frame
    x = np.full(frame_total, np.nan, dtype=float)
    y = np.full(frame_total, np.nan, dtype=float)

    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    previous: tuple[float, float] | None = None
    last_percent = -1
    for offset in range(frame_total):
        ok, frame = capture.read()
        if not ok:
            break
        center = find_blue_center(frame, arena, previous)
        if center is not None:
            x[offset], y[offset] = center
            previous = center
        percent = int(100 * (offset + 1) / frame_total)
        if percent // 10 != last_percent // 10:
            progress(f"{path.name}: tracking {percent}%")
            last_percent = percent
    capture.release()

    raw_valid = np.isfinite(x) & np.isfinite(y)
    maximum_gap = max(1, int(round(0.5 * info.fps)))
    x = interpolate_short_gaps(x, maximum_gap)
    y = interpolate_short_gaps(y, maximum_gap)
    x = rolling_nanmedian(x, 5)
    y = rolling_nanmedian(y, 5)
    return x, y, raw_valid, info, start_frame, end_frame


def dotted_circle(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    for angle in range(0, 360, 8):
        radians = math.radians(angle)
        point = (
            int(round(center[0] + radius * math.cos(radians))),
            int(round(center[1] + radius * math.sin(radians))),
        )
        cv2.circle(image, point, max(2, thickness), color, -1, lineType=cv2.LINE_AA)


def save_trajectory_tiff(
    output: Path,
    background: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    arena: Arena,
) -> None:
    canvas = cv2.addWeighted(background, 0.35, np.full_like(background, 255), 0.65, 0)
    valid = np.isfinite(x) & np.isfinite(y)
    points = np.column_stack((x[valid], y[valid])).round().astype(np.int32)
    if len(points) >= 2:
        # Break the trace at large jumps so missing detections do not create chords.
        for first, second in zip(points[:-1], points[1:]):
            if np.linalg.norm(second - first) <= arena.radius_px * 0.20:
                cv2.line(canvas, tuple(first), tuple(second), (180, 80, 0), 2, cv2.LINE_AA)

    center = (int(round(arena.center_x_px)), int(round(arena.center_y_px)))
    cv2.circle(canvas, center, int(round(arena.radius_px)), (40, 40, 40), 2, cv2.LINE_AA)
    dotted_circle(canvas, center, int(round(arena.center_radius_px)), (0, 0, 230), 3)
    label_at = (
        int(round(arena.center_x_px - arena.center_radius_px)),
        max(28, int(round(arena.center_y_px - arena.center_radius_px - 15))),
    )
    cv2.putText(
        canvas,
        "Center zone boundary",
        label_at,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, "Locomotion trajectory", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_TIFF_COMPRESSION, 5]):
        raise RuntimeError(f"Could not save TIFF: {output}")


def zone_distance(step_mm: np.ndarray, mask: np.ndarray) -> float:
    return float(np.nansum(np.where(mask, step_mm, np.nan)))


def analyze_one(
    path: Path,
    output_dir: Path,
    start_min: float,
    end_min: float,
    pixels_per_mm: float,
    progress: Callable[[str], None],
) -> dict[str, object]:
    progress(f"{path.name}: detecting arena")
    arena, background = detect_arena(path)
    x, y, raw_valid, info, start_frame, end_frame = track_video(
        path, arena, start_min * 60.0, end_min * 60.0, progress
    )
    valid = np.isfinite(x) & np.isfinite(y)
    radial = np.hypot(x - arena.center_x_px, y - arena.center_y_px)
    in_center = valid & (radial <= arena.center_radius_px)
    in_peripheral = valid & ~in_center

    step_mm = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)) / pixels_per_mm
    adjacent_valid = valid & np.concatenate(([False], valid[:-1]))
    step_mm[~adjacent_valid] = np.nan
    # Remove only obvious tracking errors; this is intentionally much higher
    # than ordinary mouse locomotion and does not define an immobility threshold.
    speed_mm_s = step_mm * info.fps
    step_mm[speed_mm_s > 2000.0] = np.nan

    center_time_s = float(in_center.sum() / info.fps)
    peripheral_time_s = float(in_peripheral.sum() / info.fps)
    center_distance_mm = zone_distance(step_mm, in_center)
    peripheral_distance_mm = zone_distance(step_mm, in_peripheral)
    total_distance_mm = center_distance_mm + peripheral_distance_mm

    comparable = valid[1:] & valid[:-1]
    center_entries = int(np.sum(comparable & in_center[1:] & ~in_center[:-1]))

    trajectory = output_dir / f"{path.stem}_locomotion_trajectory.tiff"
    save_trajectory_tiff(trajectory, background, x, y, arena)
    analyzed_duration_s = (end_frame - start_frame) / info.fps
    missing_time_s = float((~raw_valid).sum() / info.fps)
    return {
        "video": path.name,
        "start_time_min": start_frame / info.fps / 60.0,
        "end_time_min": end_frame / info.fps / 60.0,
        "analyzed_duration_s": analyzed_duration_s,
        "total_walking_distance_mm": total_distance_mm,
        "average_speed_all_analyzed_duration_mm_s": (
            total_distance_mm / analyzed_duration_s if analyzed_duration_s else math.nan
        ),
        "center_zone_walking_distance_mm": center_distance_mm,
        "center_zone_time_s": center_time_s,
        "center_zone_mean_speed_mm_s": center_distance_mm / center_time_s if center_time_s else math.nan,
        "peripheral_region_walking_distance_mm": peripheral_distance_mm,
        "peripheral_region_time_s": peripheral_time_s,
        "peripheral_region_mean_speed_mm_s": peripheral_distance_mm / peripheral_time_s if peripheral_time_s else math.nan,
        "total_crossings_into_center_zone": center_entries,
        "missing_time_s": missing_time_s,
    }


def collect_videos(location: Path) -> list[Path]:
    if location.is_file():
        if location.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video type: {location.suffix}")
        return [location.resolve()]
    if not location.is_dir():
        raise FileNotFoundError(location)
    videos = sorted(
        path.resolve()
        for path in location.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and "labeled" not in path.stem.lower()
    )
    if not videos:
        raise FileNotFoundError(f"No video files found in {location}")
    return videos


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("There are no successful results to save")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    location: Path,
    start_min: float,
    end_min: float,
    pixels_per_mm: float,
    output_dir: Path | None = None,
    progress: Callable[[str], None] = print,
) -> tuple[Path, list[Path], list[str]]:
    if start_min < 0 or end_min <= start_min:
        raise ValueError("End time must be greater than start time")
    if pixels_per_mm <= 0:
        raise ValueError("Pixels/mm must be positive")
    videos = collect_videos(location.expanduser())
    destination = output_dir or (location if location.is_dir() else location.parent)
    destination.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    failures: list[str] = []
    tiffs: list[Path] = []
    video_count = len(videos)
    progress(f"Batch contains {video_count} video(s)")
    for video_number, video in enumerate(videos, start=1):
        try:
            progress(f"Batch video {video_number}/{video_count}: {video.name}")
            result = analyze_one(video, destination, start_min, end_min, pixels_per_mm, progress)
            results.append(result)
            tiffs.append(destination / f"{video.stem}_locomotion_trajectory.tiff")
        except Exception as exc:
            failures.append(f"{video.name}: {type(exc).__name__}: {exc}")
    if not results:
        raise RuntimeError("No videos were successfully analyzed. " + " | ".join(failures))
    outpath = destination / "OFT_locomotion_results.csv"
    write_csv(outpath, results)
    print(f"\n✅ Summary saved to: {outpath}")
    progress(f"Batch complete: {len(results)}/{video_count} video(s) analyzed successfully")
    return outpath, tiffs, failures


def graphical_interface() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError:
        print("Graphical window support is unavailable in this Python installation.")
        location_text = input(f"Video or folder [{DEFAULT_LOCATION}]: ").strip()
        start_text = input("Start time in minutes [1]: ").strip()
        end_text = input("End time in minutes [11]: ").strip()
        scale_text = input("Scale in pixels/mm [2.7]: ").strip()
        csv_path, tiffs, failures = run_analysis(
            Path(location_text or DEFAULT_LOCATION),
            float(start_text or 1.0),
            float(end_text or 11.0),
            float(scale_text or 2.7),
        )
        print(f"CSV: {csv_path}")
        for path in tiffs:
            print(f"Trajectory: {path}")
        if failures:
            print("Failures:", file=sys.stderr)
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
            return 1
        return 0

    root = tk.Tk()
    root.title("Mouse Open Field Locomotion Analysis")
    root.resizable(False, False)
    location_var = tk.StringVar(value=str(DEFAULT_LOCATION))
    start_var = tk.StringVar(value="1")
    end_var = tk.StringVar(value="11")
    scale_var = tk.StringVar(value="2.7")
    status_var = tk.StringVar(value="Choose a video or folder, then click Analyze.")

    frame = ttk.Frame(root, padding=18)
    frame.grid(row=0, column=0, sticky="nsew")
    ttk.Label(frame, text="Video or folder location").grid(row=0, column=0, sticky="w")
    ttk.Entry(frame, textvariable=location_var, width=62).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(3, 9))

    def choose_video() -> None:
        selected = filedialog.askopenfilename(
            initialdir=str(DEFAULT_LOCATION),
            filetypes=[("Video files", "*.mov *.MOV *.mp4 *.MP4 *.avi *.AVI *.m4v"), ("All files", "*")],
        )
        if selected:
            location_var.set(selected)

    def choose_folder() -> None:
        selected = filedialog.askdirectory(initialdir=str(DEFAULT_LOCATION))
        if selected:
            location_var.set(selected)

    ttk.Button(frame, text="Choose video", command=choose_video).grid(row=2, column=0, sticky="w")
    ttk.Button(frame, text="Choose folder", command=choose_folder).grid(row=2, column=1, sticky="w")

    ttk.Label(frame, text="Start time (minutes)").grid(row=3, column=0, sticky="w", pady=(14, 0))
    ttk.Label(frame, text="End time (minutes)").grid(row=3, column=1, sticky="w", pady=(14, 0))
    ttk.Label(frame, text="Scale (pixels/mm)").grid(row=3, column=2, sticky="w", pady=(14, 0))
    ttk.Entry(frame, textvariable=start_var, width=15).grid(row=4, column=0, sticky="w")
    ttk.Entry(frame, textvariable=end_var, width=15).grid(row=4, column=1, sticky="w")
    ttk.Entry(frame, textvariable=scale_var, width=15).grid(row=4, column=2, sticky="w")
    ttk.Label(frame, textvariable=status_var, wraplength=580).grid(row=5, column=0, columnspan=3, sticky="w", pady=(16, 10))

    def update_status(message: str) -> None:
        status_var.set(message)
        root.update_idletasks()

    def analyze() -> None:
        try:
            analyze_button.state(["disabled"])
            csv_path, tiffs, failures = run_analysis(
                Path(location_var.get()),
                float(start_var.get()),
                float(end_var.get()),
                float(scale_var.get()),
                progress=update_status,
            )
            message = f"Finished. CSV: {csv_path}\nTrajectory TIFF files: {len(tiffs)}"
            if failures:
                message += "\n\nSome videos failed:\n" + "\n".join(failures)
            status_var.set(message)
            messagebox.showinfo("Analysis complete", message)
        except Exception as exc:
            status_var.set(f"Error: {exc}")
            messagebox.showerror("Analysis error", str(exc))
        finally:
            analyze_button.state(["!disabled"])

    analyze_button = ttk.Button(frame, text="Analyze", command=analyze)
    analyze_button.grid(row=6, column=0, sticky="w")
    ttk.Button(frame, text="Close", command=root.destroy).grid(row=6, column=1, sticky="w")
    root.mainloop()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("location", nargs="?", type=Path, help="Video file or folder of videos")
    parser.add_argument(
        "--folder",
        type=Path,
        help="Folder for batch analysis, or one video file",
    )
    parser.add_argument(
        "--start", "--start-min", dest="start_min", type=float, default=1.0,
        help="Analysis start time in minutes (default: 1)",
    )
    parser.add_argument(
        "--end", "--end-min", dest="end_min", type=float, default=11.0,
        help="Analysis end time in minutes (default: 11)",
    )
    parser.add_argument(
        "--mm", "--pixels-per-mm", dest="pixels_per_mm", type=float, default=2.7,
        help="Scale in pixels per millimeter (default: 2.7)",
    )
    parser.add_argument("--output-dir", type=Path, help="Optional output directory")
    args = parser.parse_args()
    if args.location is not None and args.folder is not None:
        parser.error("Use either a positional location or --folder, not both")
    return args


def main() -> int:
    args = parse_args()
    location = args.folder or args.location
    if location is None:
        return graphical_interface()
    csv_path, tiffs, failures = run_analysis(
        location,
        args.start_min,
        args.end_min,
        args.pixels_per_mm,
        args.output_dir,
    )
    print(f"CSV: {csv_path}")
    for path in tiffs:
        print(f"Trajectory: {path}")
    if failures:
        print("Failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
