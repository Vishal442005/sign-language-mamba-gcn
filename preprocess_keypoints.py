"""
preprocess_keypoints.py
=======================
Offline MediaPipe hand-keypoint extraction for the ASL Alphabet dataset.

Usage (run once before training):
    python preprocess_keypoints.py --data_root ./dataset_extracted/asl_alphabet_train/asl_alphabet_train

For each image found under <data_root>/<class>/<image>.jpg this script:
  1. Runs MediaPipe Hands to detect 21 3-D landmarks.
  2. Normalises the landmarks:
       - Translate so wrist (landmark 0) is at origin.
       - Scale by the max L2 distance from wrist (scale-invariant).
  3. Saves a (21, 3) float32 numpy array as <image_stem>.npy alongside the image.
  4. When no hand is detected a zero array is saved as fallback.

The cache files are consumed by SignLanguageDualModalDataset in dataset.py.
"""

import argparse
import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm


# ---------------------------------------------------------------------------
# MediaPipe initialisation
# ---------------------------------------------------------------------------
mp_hands = mp.solutions.hands


def extract_keypoints(image_bgr: np.ndarray, hands_detector) -> np.ndarray:
    """Return a (21, 3) float32 array of normalised hand landmarks.

    Normalisation:
        - Wrist-centred: subtract landmark[0] (wrist) from all landmarks.
        - Scale-invariant: divide by max L2 distance from wrist (clamped ≥ 1e-6).

    Returns a zero array if no hand is detected.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)

    kpts = np.zeros((21, 3), dtype=np.float32)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]  # use the first detected hand
        for i, lm in enumerate(hand.landmark):
            kpts[i] = [lm.x, lm.y, lm.z]

        # Wrist-centre normalisation
        wrist = kpts[0].copy()
        kpts -= wrist

        # Scale-invariant normalisation
        distances = np.linalg.norm(kpts, axis=1)  # (21,)
        scale = distances.max()
        if scale < 1e-6:
            scale = 1e-6
        kpts /= scale

    return kpts


def process_directory(data_root: str, max_num_hands: int = 1,
                      detection_confidence: float = 0.5) -> None:
    """Walk <data_root>/<class>/<image> and write .npy keypoint caches."""

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # Collect all image paths
    image_paths = []
    for root, dirs, files in os.walk(data_root):
        dirs.sort()
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, fname))

    print(f"Found {len(image_paths)} images under '{data_root}'")

    skipped = 0
    no_hand = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=max_num_hands,
        min_detection_confidence=detection_confidence,
    ) as hands:
        for img_path in tqdm(image_paths, desc="Extracting keypoints", unit="img"):
            npy_path = os.path.splitext(img_path)[0] + '.npy'

            # Skip if already processed (re-run safe)
            if os.path.exists(npy_path):
                skipped += 1
                continue

            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                # Save zeros for unreadable images
                np.save(npy_path, np.zeros((21, 3), dtype=np.float32))
                no_hand += 1
                continue

            kpts = extract_keypoints(image_bgr, hands)

            if np.all(kpts == 0):
                no_hand += 1

            np.save(npy_path, kpts)

    total = len(image_paths)
    processed = total - skipped
    print(f"\nDone. Processed: {processed} | Skipped (cached): {skipped} | "
          f"No-hand detections (zero fallback): {no_hand}/{processed}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline MediaPipe keypoint extraction for ASL Alphabet dataset"
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='./dataset_extracted/asl_alphabet_train/asl_alphabet_train',
        help='Root directory of the dataset (class folders inside)'
    )
    parser.add_argument(
        '--max_num_hands',
        type=int,
        default=1,
        help='Max number of hands for MediaPipe (default: 1)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='MediaPipe min_detection_confidence (default: 0.5)'
    )
    args = parser.parse_args()

    process_directory(
        data_root=args.data_root,
        max_num_hands=args.max_num_hands,
        detection_confidence=args.confidence,
    )


if __name__ == '__main__':
    main()
