import argparse
import hashlib
import os
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Tuple

from PIL import Image, ExifTags
import imagehash
from tqdm import tqdm
import csv

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".heic"}

def get_exif_title(img: Image.Image) -> Optional[str]:
    """
    Try to read a human-entered title from EXIF.
    Common fields: ImageDescription, XPTitle (Windows), or Title (less common).
    """
    title = None
    try:
        exif = img.getexif()
        if exif:
            # Map numeric tags to names
            tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            # ImageDescription (string)
            cand = tag_map.get("ImageDescription")
            if isinstance(cand, bytes):
                try:
                    cand = cand.decode("utf-8", errors="ignore")
                except Exception:
                    cand = None
            if cand:
                title = cand.strip() or None

            # XPTitle (bytes of UTF-16-LE with nulls)
            if not title and "XPTitle" in tag_map:
                xp = tag_map["XPTitle"]
                if isinstance(xp, bytes):
                    try:
                        # XP fields often are UTF-16-LE with trailing nulls
                        cand = xp.decode("utf-16-le", errors="ignore").rstrip("\x00")
                        title = cand.strip() or None
                    except Exception:
                        pass

            # Some software writes "Title"
            if not title and "Title" in tag_map:
                cand = tag_map["Title"]
                if isinstance(cand, bytes):
                    try:
                        cand = cand.decode("utf-8", errors="ignore")
                    except Exception:
                        cand = None
                if cand:
                    title = cand.strip() or None
    except Exception:
        pass
    return title

def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def image_phash(path: Path) -> Optional[imagehash.ImageHash]:
    try:
        with Image.open(path) as im:
            im.load()
            return imagehash.phash(im)
    except Exception:
        return None

def image_title(path: Path) -> str:
    # Prefer EXIF title; fall back to filename stem
    try:
        with Image.open(path) as im:
            t = get_exif_title(im)
            if t:
                return t
    except Exception:
        pass
    return path.stem

def find_images(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return files

def group_exact_duplicates(paths: List[Path]) -> dict:
    by_md5 = defaultdict(list)
    for p in tqdm(paths, desc="Hashing (exact)"):
        try:
            h = file_md5(p)
            by_md5[h].append(p)
        except Exception:
            # unreadable file; skip
            pass
    return {k: v for k, v in by_md5.items() if len(v) > 1}

def group_near_duplicates(paths: List[Path], threshold: int) -> List[List[Path]]:
    """
    Clusters images whose perceptual hash distance <= threshold.
    Simple single-link (union-find-like) clustering.
    """
    # Compute phashes
    items: List[Tuple[Path, Optional[imagehash.ImageHash]]] = []
    for p in tqdm(paths, desc="Hashing (perceptual)"):
        items.append((p, image_phash(p)))

    # Filter out failed hashes
    items = [(p, h) for p, h in items if h is not None]

    # Disjoint-set (union-find)
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # Compare all pairs (O(n^2)); fine for a few thousand images.
    # For very large sets, consider LSH/bucketing by hash value.
    for i in tqdm(range(len(items)), desc="Comparing (perceptual)"):
        hi = items[i][1]
        for j in range(i + 1, len(items)):
            hj = items[j][1]
            if hi - hj <= threshold:
                union(i, j)

    # Collect clusters
    clusters = defaultdict(list)
    for idx, (p, _) in enumerate(items):
        clusters[find(idx)].append(p)

    # Keep only clusters with >1 item
    return [v for v in clusters.values() if len(v) > 1]

def print_groups(
    exact_groups: dict,
    near_groups: List[List[Path]],
    out_csv: Optional[Path] = None,
):
    printed_any = False

    def rows_for_group(group_paths: List[Path], kind: str):
        for p in sorted(group_paths):
            yield {
                "group_type": kind,
                "title": image_title(p),
                "file": str(p),
            }

    if exact_groups:
        printed_any = True
        print("\n=== Exact duplicates (byte-for-byte) ===")
        for md5, group in exact_groups.items():
            print(f"\nGroup MD5 {md5}:")
            for p in sorted(group):
                print(f"  - {image_title(p)}  [{p.name}]  -> {p}")

    # Deduplicate near groups by ignoring groups fully contained in an exact group
    exact_sets = [set(v) for v in exact_groups.values()]
    filtered_near = []
    for g in near_groups:
        s = set(g)
        if any(s.issubset(es) for es in exact_sets):
            continue
        filtered_near.append(g)

    if filtered_near:
        printed_any = True
        print("\n=== Near-duplicates (visual similarity) ===")
        for idx, group in enumerate(filtered_near, 1):
            print(f"\nCluster #{idx}:")
            for p in sorted(group):
                print(f"  - {image_title(p)}  [{p.name}]  -> {p}")

    if not printed_any:
        print("No duplicates found.")

    if out_csv:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["group_type", "title", "file"])
            writer.writeheader()
            for md5, group in exact_groups.items():
                for row in rows_for_group(group, "exact"):
                    writer.writerow(row)
            for group in filtered_near:
                for row in rows_for_group(group, "near"):
                    writer.writerow(row)
        print(f"\nCSV report written to: {out_csv}")

def main():
    parser = argparse.ArgumentParser(
        description="Scan a folder for duplicate images and print their titles."
    )
    parser.add_argument("folder", type=str, help="Path to the image folder")
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=6,
        help="Perceptual hash Hamming distance threshold (0=only very close matches; typical 4-10).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write a CSV report of duplicate groups.",
    )
    args = parser.parse_args()

    root = Path(args.folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Folder not found or not a directory: {root}")

    images = find_images(root)
    if not images:
        print("No images found.")
        return

    exact_groups = group_exact_duplicates(images)
    near_groups = group_near_duplicates(images, threshold=args.phash_threshold)

    out_csv = Path(args.out) if args.out else None
    print_groups(exact_groups, near_groups, out_csv=out_csv)

if __name__ == "__main__":
    main()
