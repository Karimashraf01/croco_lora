# Copyright (C) 2022-present Naver Corporation.
# Relative pose evaluation using CroCo features
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Compose
from models.croco import CroCoNet
import math
from peft import PeftModel

# ------------------------------------------------------------
# Utility: Quaternion to rotation matrix
# ------------------------------------------------------------
def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


# ------------------------------------------------------------
# Load COLMAP poses (images.txt)
# ------------------------------------------------------------
def load_colmap_poses(images_txt_path):
    poses = {}

    with open(images_txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("#"):
            continue

        elems = line.split()
        if len(elems) < 10:
            continue

        qw, qx, qy, qz = map(float, elems[1:5])
        tx, ty, tz = map(float, elems[5:8])
        name = elems[9]

        R = qvec2rotmat([qw, qx, qy, qz])
        t = np.array([tx, ty, tz]).reshape(3, 1)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3:] = t

        poses[name] = T

    return poses


# ------------------------------------------------------------
# Extract encoder patch features
# ------------------------------------------------------------
def extract_features(model, image):
    with torch.inference_mode():
        # tokens = model.encoder(image)  # [1, N, D]
        feat, pos, mask = model._encode_image(image, do_mask=False)

    feat = feat[0]  # remove batch dimension

    N, D = feat.shape
    H = W = int(math.sqrt(N))

    feat = feat.reshape(H, W, D)

    return feat.cpu()

# ------------------------------------------------------------
# Dense mutual nearest neighbor matching
# ------------------------------------------------------------
def dense_match(feat1, feat2, patch_size, sim_threshold=0.5):

    H, W, D = feat1.shape

    f1 = feat1.reshape(-1, D)
    f2 = feat2.reshape(-1, D)

    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)

    sim = torch.mm(f1, f2.t())  # cosine similarity matrix

    nn12 = sim.argmax(dim=1)
    nn21 = sim.argmax(dim=0)

    pts1 = []
    pts2 = []

    for i, j in enumerate(nn12):

        # similarity score of best match
        score = sim[i, j].item()

        # Mutual nearest neighbor + threshold
        if nn21[j] == i and score > sim_threshold:

            row1 = i // W
            col1 = i % W

            row2 = j.item() // W
            col2 = j.item() % W

            x1 = (col1 + 0.5) * patch_size
            y1 = (row1 + 0.5) * patch_size

            x2 = (col2 + 0.5) * patch_size
            y2 = (row2 + 0.5) * patch_size

            pts1.append([x1, y1])
            pts2.append([x2, y2])

    return np.array(pts1), np.array(pts2)

# ------------------------------------------------------------
# Pose estimation
# ------------------------------------------------------------
def estimate_pose(pts1, pts2, K):

    if len(pts1) < 8:
        return None, None

    E, mask = cv2.findEssentialMat(
        pts1, pts2,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:
        return None, None

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    return R, t


# ------------------------------------------------------------
# Relative pose error
# ------------------------------------------------------------
def relative_pose_error(T1, T2, R_pred, t_pred):

    T_gt = T2 @ np.linalg.inv(T1)
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]

    # rotation error
    R_diff = R_pred @ R_gt.T
    angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
    rot_error = np.degrees(angle)

    # translation direction error
    t_pred = t_pred.flatten() / np.linalg.norm(t_pred)
    t_gt = t_gt / np.linalg.norm(t_gt)

    trans_error = np.degrees(
        np.arccos(np.clip(np.dot(t_pred, t_gt), -1, 1))
    )

    return rot_error, trans_error


def draw_patch_matches(
        img1_path,
        img2_path,
        pts1,
        pts2,
        croco_input_size=224,
        max_matches=1000,
        save_path=None):

    # Load original images
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))

    if img1 is None or img2 is None:
        raise ValueError("Could not load images")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Scale factors (from CroCo resized space -> original)
    sx1 = w1 / croco_input_size
    sy1 = h1 / croco_input_size

    sx2 = w2 / croco_input_size
    sy2 = h2 / croco_input_size

    # Create side-by-side canvas
    canvas = np.hstack([img1, img2])

    # Limit number of matches for visualization
    if len(pts1) > max_matches:
        idx = np.random.choice(len(pts1), max_matches, replace=False)
        pts1 = pts1[idx]
        pts2 = pts2[idx]

    for (x1, y1), (x2, y2) in zip(pts1, pts2):

        # Scale back to original resolution
        x1_orig = int(x1 * sx1)
        y1_orig = int(y1 * sy1)

        x2_orig = int(x2 * sx2)
        y2_orig = int(y2 * sy2)

        # Shift x2 because second image is on the right
        x2_shifted = x2_orig + w1

        color = tuple(np.random.randint(0, 255, 3).tolist())

        cv2.circle(canvas, (x1_orig, y1_orig), 4, color, -1)
        cv2.circle(canvas, (x2_shifted, y2_orig), 4, color, -1)

        cv2.line(canvas,
                 (x1_orig, y1_orig),
                 (x2_shifted, y2_orig),
                 color, 1)

    if save_path is not None:
        cv2.imwrite(save_path, canvas)

    return canvas

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    trfs = Compose([
        ToTensor(),
        Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Load images
    img1_name = r'labeled_videos\00444043_ES04\frames\img_000000.jpg'
    img2_name = r'labeled_videos\00444043_ES04\frames\img_000001.jpg'
    image1 = trfs(Image.open(img1_name).convert('RGB').resize((224,224))).to(device, non_blocking=True).unsqueeze(0)
    image2 = trfs(Image.open(img2_name).convert('RGB').resize((224,224))).to(device, non_blocking=True).unsqueeze(0)

    # Load CroCo
    ckpt = torch.load('pretrained_models/CroCo_V2_ViTBase_SmallDecoder.pth', 'cpu')

    #  Rebuild architecture
    model = CroCoNet(**ckpt["croco_kwargs"]).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    model.to(device).eval()

    model.eval()


    # Extract features
    feat1 = extract_features(model, image1)
    feat2 = extract_features(model, image2)
    print("Feature shapes:", feat1.shape, feat2.shape)

    patch_size = 16
    pts1, pts2 = dense_match(feat1, feat2, patch_size)

    draw_patch_matches(img1_name, img2_name, pts1, pts2, save_path="matches.jpg")

    print("Number of matches:", len(pts1))

    # # Example camera intrinsics (replace with scaled COLMAP intrinsics!)
    fx = fy = 1000
    cx = cy = 112  # center of 224
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

    R_pred, t_pred = estimate_pose(pts1, pts2, K)

    if R_pred is None:
        print("Pose estimation failed")
        return
    print("Estimated rotation:\n", R_pred)
    print("Estimated translation:\n", t_pred)

if __name__ == "__main__":
    main()