import os
import random
import cv2
import numpy as np
from collections import defaultdict


def generate_sequential_pairs(
    images_dir: str,
    output_txt: str,
    reverse_percentage: float = 0.0,
    full_paths: bool = True,
    save_viz_sample: bool = False,
    viz_output_path: str = "pairs_visualization.jpg",
    num_viz_samples: int = 10,
):
    """
    Generate sequential CroCo pairs from images and optionally save visualization.

    Args:
        images_dir (str): Path to directory containing images.
        output_txt (str): Path to save the pairs file.
        reverse_percentage (float): Percentage (0-100) of pairs to reverse.
        full_paths (bool): Write full paths or filenames only.
        save_viz_sample (bool): Save a stacked visualization sample.
        viz_output_path (str): Output image path for visualization.
        num_viz_samples (int): Number of pairs to visualize.
    """

    if not os.path.isdir(images_dir):
        raise ValueError(f"Directory not found: {images_dir}")

    if not (0.0 <= reverse_percentage <= 100.0):
        raise ValueError("reverse_percentage must be between 0 and 100")

    scenes = defaultdict(list)

    # ---- Group by scene ----
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        name = os.path.splitext(fname)[0]
        parts = name.split("_")

        if len(parts) < 2:
            continue

        scene_id = parts[0]
        frame_id = int(parts[-1])

        scenes[scene_id].append((frame_id, fname))

    # ---- Sort frames ----
    for scene_id in scenes:
        scenes[scene_id] = sorted(scenes[scene_id], key=lambda x: x[0])

    # ---- Generate forward pairs ----
    pairs = []

    for scene_id, frames in scenes.items():
        for i in range(len(frames) - 1):
            img1 = frames[i][1]
            img2 = frames[i + 1][1]

            if full_paths:
                img1 = os.path.join(images_dir, img1)
                img2 = os.path.join(images_dir, img2)

            pairs.append((img1, img2))

    print(f"Generated {len(pairs)} forward pairs.")

    # ---- Add reversed pairs ----
    if reverse_percentage > 0 and len(pairs) > 0:
        num_reverse = int(len(pairs) * (reverse_percentage / 100.0))
        reverse_samples = random.sample(pairs, num_reverse)
        reversed_pairs = [(b, a) for (a, b) in reverse_samples]
        pairs.extend(reversed_pairs)
        print(f"Added {len(reversed_pairs)} reversed pairs.")

    # ---- Save pairs file ----
    with open(output_txt, "w") as f:
        for img1, img2 in pairs:
            f.write(f"{img1} {img2}\n")

    print(f"Saved total {len(pairs)} pairs to {output_txt}")

    # ---- Visualization ----
    if save_viz_sample and len(pairs) > 0:
        _save_visualization_sample(
            pairs,
            viz_output_path,
            num_viz_samples,
        )

    return pairs


def _save_visualization_sample(pairs, output_path, num_samples):
    """Save stacked visualization of sample pairs (robust version)."""

    sample_pairs = random.sample(
        pairs, min(num_samples, len(pairs))
    )

    stacked_rows = []

    for img1_path, img2_path in sample_pairs:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            continue

        # Resize to same height within pair
        h = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
        img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))

        pair_img = np.hstack([img1, img2])
        stacked_rows.append(pair_img)

    if len(stacked_rows) == 0:
        print("No valid images for visualization.")
        return

    # ---- Pad all rows to same width ----
    max_width = max(img.shape[1] for img in stacked_rows)

    padded_rows = []
    for img in stacked_rows:
        h, w, c = img.shape
        if w < max_width:
            pad_width = max_width - w
            padding = np.zeros((h, pad_width, 3), dtype=np.uint8)
            img = np.hstack([img, padding])
        padded_rows.append(img)

    final_image = np.vstack(padded_rows)

    cv2.imwrite(output_path, final_image)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    generate_sequential_pairs(
    images_dir="New_drone_dataset\VisDrone2019-DET-test-dev\images",
    output_txt="New_drone_dataset/croco_pairs_test_dev.txt",
    reverse_percentage=30,  # add reverse for 30% of pairs
    full_paths=True,
    save_viz_sample=True,
    viz_output_path="New_drone_dataset/viz_sample_test_dev.jpg",
)
    

## todo: 
# check lighting conditiions
# add place regcognition model to ensure gaps beteen pairs
# add option to control the gap between frames 