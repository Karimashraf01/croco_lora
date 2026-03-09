import cv2
import numpy as np
from pathlib import Path


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
GAP = 30   # frame gap (baseline control)

paths = [
    Path("00444043_ES04"),
    Path("00444043_ES05"),
    Path("3010040645-DM4K_es03"),
    Path("c1dwee7v81b5don4ak86vtcu_512"),
    Path("c1dwee7v81b5don4ak86vtcu_692"),
    Path("c2d7b15a4kz60ol65894uwiu"),
    Path("c8jka0cpama5ykz5ybc6aj3u"),
    Path("cc4nowntl76xocrpx6eqj8cu_0"),
    Path("ch5ad04a78eqdtkge4tu0jdu"),
    Path("cskpnp1m39g9b110ri229wyu_10s"),
    Path("WM_021024_THERIVER_06_0014_WM_021024_THERIVER_06_0014_es01")
]


# ------------------------------------------------------------
# quaternion → rotation matrix
# ------------------------------------------------------------
def qvec2rotmat(q):

    w, x, y, z = q

    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


# ------------------------------------------------------------
# read camera intrinsics
# ------------------------------------------------------------
def read_camera(camera_path):

    with open(camera_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            return line.strip()


# ------------------------------------------------------------
# read COLMAP poses
# ------------------------------------------------------------
def read_images(images_path):

    poses = []
    names = []

    with open(images_path) as f:

        for line in f:

            if line.startswith("#"):
                continue

            elems = line.split()

            if len(elems) < 10:
                continue

            q = np.array(list(map(float, elems[1:5])))
            t = np.array(list(map(float, elems[5:8])))

            name = elems[9]

            R = qvec2rotmat(q)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t

            poses.append(T)
            names.append(name)

    return poses, names


# ------------------------------------------------------------
# extract frames from video
# ------------------------------------------------------------
def extract_frames(video_path, frames_dir, names):

    cap = cv2.VideoCapture(str(video_path))

    for i, name in enumerate(names):

        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite(str(frames_dir / name), frame)

    cap.release()

    print(f"Extracted {len(names)} frames")


# ------------------------------------------------------------
# relative pose
# ------------------------------------------------------------
def relative_pose(T1, T2):

    return T2 @ np.linalg.inv(T1)


# ------------------------------------------------------------
# write dataset pairs
# ------------------------------------------------------------
def write_pairs(output_txt, poses, names, camera_line, gap):

    with open(output_txt, "w") as f:

        # write camera intrinsics
        f.write(camera_line + "\n")

        for i in range(0,len(poses) - gap,gap):

            T1 = poses[i]
            T2 = poses[i + gap]

            rel = relative_pose(T1, T2)

            r = rel[:3, :3].reshape(-1)
            t = rel[:3, 3]

            pose = np.concatenate([r, t])

            f.write(
                f"frames/{names[i]} frames/{names[i+gap]} "
                + " ".join(map(str, pose))
                + "\n"
            )

    print("Saved pairs:", output_txt)


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
for ROOT in paths:

    print("\nProcessing:", ROOT)

    VIDEO_PATH = list(ROOT.glob("*.mp4"))[0]

    IMAGES_TXT = ROOT / "gt" / "colmap" / "images.txt"
    CAMERAS_TXT = ROOT / "gt" / "colmap" / "cameras.txt"

    FRAMES_DIR = ROOT / "frames"
    OUTPUT_TXT = ROOT / "pairs_with_pose.txt"

    FRAMES_DIR.mkdir(exist_ok=True)

    # load camera
    camera_line = read_camera(CAMERAS_TXT)

    # load poses
    poses, names = read_images(IMAGES_TXT)

    print("Frames in COLMAP:", len(names))

    # extract frames
    extract_frames(VIDEO_PATH, FRAMES_DIR, names)

    # generate pairs
    write_pairs(OUTPUT_TXT, poses, names, camera_line, GAP)

print("\nAll datasets processed.")