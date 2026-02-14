import sys
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def list_projects(base_dir: Path) -> list[Path]:
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    return sorted([p for p in base_dir.iterdir() if p.is_dir()])


def prompt_for_project(projects: list[Path]) -> str | None:
    if not projects:
        return None
    print("Available projects:")
    for idx, proj in enumerate(projects, start=1):
        print(f"{idx}. {proj.name}")
    raw = input("Select a project by number or name: ").strip()
    if not raw:
        return None
    if raw.isdigit():
        num = int(raw)
        if 1 <= num <= len(projects):
            return projects[num - 1].name
        return None
    return raw


def iter_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    return sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def prompt_range(total: int) -> tuple[int, int] | None:
    print(f"Total images available: {total}")
    raw_start = input("Start index (1-based, default 1): ").strip() or "1"
    raw_end = input(f"End index (1-based, default {total}): ").strip() or str(total)
    if not (raw_start.isdigit() and raw_end.isdigit()):
        return None
    start = int(raw_start)
    end = int(raw_end)
    if start < 1 or end < 1 or start > end or end > total:
        return None
    return start, end


def prompt_yes_no(message: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    raw = input(message + suffix).strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def prompt_outputs() -> tuple[bool, bool, bool]:
    print("Select outputs to generate:")
    startrail_image = prompt_yes_no("Generate startrail image?", True)
    startrail_video = prompt_yes_no("Generate startrail video?", True)
    starmotion_video = prompt_yes_no("Generate starmotion video?", True)
    return startrail_image, startrail_video, starmotion_video


def prompt_duration_seconds(default: float = 10.0) -> float | None:
    raw = input(f"Video duration in seconds (default {default}): ").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def resize_max_dim(img, max_dim: int | None):
    if not max_dim:
        return img
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / float(longest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_params(ext: str) -> list[int]:
    ext = ext.lower()
    if ext in {".jpg", ".jpeg"}:
        return [cv2.IMWRITE_JPEG_QUALITY, 85]
    if ext == ".png":
        return [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if ext == ".webp":
        return [cv2.IMWRITE_WEBP_QUALITY, 80]
    return []


def compress_images(
    src_files: list[Path], input_root: Path, out_dir: Path
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result_paths: list[Path] = []
    for src in src_files:
        rel = src.relative_to(input_root)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(src))
        if img is None:
            print(f"Skip (unreadable): {src}")
            continue
        params = encode_params(src.suffix)
        ok = cv2.imwrite(str(dst), img, params)
        if not ok:
            print(f"Failed: {src}")
            continue
        result_paths.append(dst)
    return result_paths


def write_videos(
    image_files: list[Path],
    output_dir: Path,
    fps: int,
    suffix: str,
    make_startrail: bool,
    make_motion: bool,
):
    first = cv2.imread(str(image_files[0]))
    h, w, _ = first.shape

    trail_video = None
    motion_video = None
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # ty:ignore[unresolved-attribute]

    if make_startrail:
        trail_video = cv2.VideoWriter(
            str(output_dir / f"star_trails{suffix}.mp4"), fourcc, fps, (w, h)
        )
    if make_motion:
        motion_video = cv2.VideoWriter(
            str(output_dir / f"star_motion{suffix}.mp4"), fourcc, fps, (w, h)
        )

    stack = None
    for idx, file in enumerate(image_files):
        img = cv2.imread(str(file))
        if img is None:
            print(f"Skip (unreadable): {file}")
            continue
        if stack is None:
            stack = img.copy()
        else:
            stack = np.maximum(stack, img)

        if trail_video is not None:
            trail_video.write(stack)
        if motion_video is not None:
            motion_video.write(img)
        print(f"Stacked frame {idx + 1}/{len(image_files)}")

    if trail_video is not None:
        trail_video.release()
        print("Video saved:", output_dir / f"star_trails{suffix}.mp4")
    if motion_video is not None:
        motion_video.release()
        print("Video saved:", output_dir / f"star_motion{suffix}.mp4")


def write_startrail_image(image_files: list[Path], output_dir: Path, suffix: str):
    stack = None
    for file in image_files:
        img = cv2.imread(str(file))
        if img is None:
            print(f"Skip (unreadable): {file}")
            continue
        if stack is None:
            stack = img.copy()
        else:
            stack = np.maximum(stack, img)

    if stack is None:
        print("No readable images for startrail.")
        return

    stack = cv2.GaussianBlur(stack, (3, 3), 0)
    out_path = output_dir / f"star_trails{suffix}.jpg"
    cv2.imwrite(str(out_path), stack)
    print("Image saved:", out_path)


def make_insta_versions(
    image_files: list[Path],
    output_dir: Path,
    fps: int,
    make_startrail: bool,
    make_motion: bool,
    make_image: bool,
):
    if not image_files:
        return

    resized_dir = output_dir / "_insta_cache"
    resized_dir.mkdir(parents=True, exist_ok=True)
    resized_files: list[Path] = []

    for src in image_files:
        img = cv2.imread(str(src))
        if img is None:
            print(f"Skip (unreadable): {src}")
            continue
        img = resize_max_dim(img, 1080)
        dst = resized_dir / src.name
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        resized_files.append(dst)

    if make_image:
        write_startrail_image(resized_files, output_dir, "_insta")
    if make_startrail or make_motion:
        write_videos(
            resized_files,
            output_dir,
            fps,
            "_insta",
            make_startrail,
            make_motion,
        )


def main() -> int:
    base_dir = Path("/projects")
    if not base_dir.exists():
        print("Base directory not found:", base_dir)
        return 1

    projects = list_projects(base_dir)
    project_name = prompt_for_project(projects)
    if not project_name:
        print("No project selected.")
        return 1

    inputs_dir = base_dir / project_name / "images"
    image_files = iter_images(inputs_dir)
    if not image_files:
        print("No images found in:", inputs_dir)
        return 1

    selected = prompt_range(len(image_files))
    if not selected:
        print("Invalid range.")
        return 1
    start, end = selected
    image_files = image_files[start - 1 : end]

    use_compressed = prompt_yes_no("Compress images first?", True)
    if use_compressed:
        compressed_dir = base_dir / project_name / "compressed"
        image_files = compress_images(image_files, inputs_dir, compressed_dir)
        if not image_files:
            print("No compressed images produced.")
            return 1

    make_image, make_startrail, make_motion = prompt_outputs()
    if not any([make_image, make_startrail, make_motion]):
        print("No outputs selected.")
        return 1

    output_dir = base_dir / project_name / "result"
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = 10
    if make_startrail or make_motion:
        duration = prompt_duration_seconds(10.0)
        if duration is None:
            print("Invalid duration.")
            return 1
        fps = max(1, int(round(len(image_files) / duration)))

    if make_image:
        write_startrail_image(image_files, output_dir, "")
    if make_startrail or make_motion:
        write_videos(image_files, output_dir, fps, "", make_startrail, make_motion)

    make_insta = prompt_yes_no("Create Instagram-compatible versions?", False)
    if make_insta:
        make_insta_versions(
            image_files,
            output_dir,
            fps,
            make_startrail,
            make_motion,
            make_image,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
