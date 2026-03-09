import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import math
import os
from torchvision.transforms import ToTensor, Normalize, Compose
from models.croco import CroCoNet
from peft import PeftModel
import argparse
import sys

# ------------------------------------------------------------
# Load dataset txt
# ------------------------------------------------------------
def load_pairs_file(txt_path):

    with open(txt_path) as f:
        lines = f.readlines()

    # camera parameters
    cam = lines[0].split()

    width = int(cam[2])
    height = int(cam[3])

    fx = float(cam[4])
    cx = float(cam[5])
    cy = float(cam[6])

    fy = fx

    K = np.array([
        [fx,0,cx],
        [0,fy,cy],
        [0,0,1]
    ])

    pairs = []

    for line in lines[1:]:

        elems = line.split()

        img1 = elems[0]
        img2 = elems[1]

        pose = np.array(list(map(float, elems[2:])))

        R = pose[:9].reshape(3,3)
        t = pose[9:]

        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t

        pairs.append((img1,img2,T))

    return K, width, height, pairs


# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------
def extract_features(model, image):

    with torch.inference_mode():

        feat, pos, mask = model._encode_image(image, do_mask=False)

    feat = feat[0]

    N,D = feat.shape
    H = W = int(math.sqrt(N))

    feat = feat.reshape(H,W,D)

    return feat.cpu()


# ------------------------------------------------------------
# Dense patch matching
# ------------------------------------------------------------
def dense_match(feat1, feat2, patch_size, sim_threshold=0.9):

    H,W,D = feat1.shape

    f1 = feat1.reshape(-1,D)
    f2 = feat2.reshape(-1,D)

    f1 = F.normalize(f1,dim=1)
    f2 = F.normalize(f2,dim=1)

    sim = torch.mm(f1,f2.t())

    nn12 = sim.argmax(dim=1)
    nn21 = sim.argmax(dim=0)

    pts1 = []
    pts2 = []

    for i,j in enumerate(nn12):

        score = sim[i,j].item()
        # print(f"Match {i} -> {j} | score={score:.4f}")

        if nn21[j] == i and score > sim_threshold:

            r1 = i // W
            c1 = i % W

            r2 = j.item() // W
            c2 = j.item() % W

            x1 = (c1 + 0.5) * patch_size
            y1 = (r1 + 0.5) * patch_size

            x2 = (c2 + 0.5) * patch_size
            y2 = (r2 + 0.5) * patch_size

            pts1.append([x1,y1])
            pts2.append([x2,y2])

    return np.array(pts1), np.array(pts2)


# ------------------------------------------------------------
# Convert points to original resolution
# ------------------------------------------------------------
def rescale_points(pts, orig_w, orig_h, croco_size=224):

    sx = orig_w / croco_size
    sy = orig_h / croco_size

    pts_scaled = pts.copy()

    pts_scaled[:,0] *= sx
    pts_scaled[:,1] *= sy

    return pts_scaled


# ------------------------------------------------------------
# Estimate pose
# ------------------------------------------------------------
def estimate_pose(pts1, pts2, K):

    if len(pts1) < 8:
        return None,None,None

    E,mask = cv2.findEssentialMat(
        pts1,
        pts2,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:
        return None,None,None

    _,R,t,mask_pose = cv2.recoverPose(E,pts1,pts2,K)


    return R,t


# ------------------------------------------------------------
# Pose error
# ------------------------------------------------------------
def pose_error(T_gt, R_pred, t_pred):

    R_gt = T_gt[:3,:3]
    t_gt = T_gt[:3,3]

    R_diff = R_pred @ R_gt.T

    angle = np.arccos(
        np.clip((np.trace(R_diff)-1)/2,-1,1)
    )

    rot_error = np.degrees(angle)

    t_pred = t_pred.flatten()
    t_pred = t_pred / np.linalg.norm(t_pred)

    t_gt = t_gt / np.linalg.norm(t_gt)

    trans_error = np.degrees(
        np.arccos(np.clip(np.dot(t_pred,t_gt),-1,1))
    )
    rot_error = min(rot_error, 180 - rot_error)
    trans_error = min(trans_error, 180 - trans_error)
    return rot_error, trans_error

# ------------------------------------------------------------
# Save match visualization
# ------------------------------------------------------------
def save_matches(img1_path, img2_path, pts1, pts2, save_path, max_matches=200):

    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))

    if img1 is None or img2 is None:
        print("Could not read images for visualization")
        return

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas = np.zeros((max(h1,h2), w1 + w2, 3), dtype=np.uint8)

    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1+w2] = img2

    # randomly subsample matches if too many
    if len(pts1) > max_matches:
        idx = np.random.choice(len(pts1), max_matches, replace=False)
        pts1 = pts1[idx]
        pts2 = pts2[idx]

    for (x1,y1),(x2,y2) in zip(pts1,pts2):

        x1 = int(x1)
        y1 = int(y1)

        x2 = int(x2) + w1
        y2 = int(y2)

        color = tuple(np.random.randint(0,255,3).tolist())

        cv2.circle(canvas,(x1,y1),4,color,-1)
        cv2.circle(canvas,(x2,y2),4,color,-1)

        cv2.line(canvas,(x1,y1),(x2,y2),color,1)
    # print(f"Saving match visualization to {save_path}")
    cv2.imwrite(str(save_path),canvas)


def print_statistics(rot_errors, trans_errors):

    rot_errors = np.array(rot_errors)
    trans_errors = np.array(trans_errors)

    print("\nRotation Error Statistics")
    print("Mean   :", np.mean(rot_errors))
    print("Median :", np.median(rot_errors))
    print("Std    :", np.std(rot_errors))

    print("\nTranslation Error Statistics")
    print("Mean   :", np.mean(trans_errors))
    print("Median :", np.median(trans_errors))
    print("Std    :", np.std(trans_errors))

def plot_error_histograms(rot_errors, trans_errors, save_path="pose_error_histograms.png", show=True):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.hist(rot_errors, bins=30)
    plt.title("Rotation Error Histogram")
    plt.xlabel("Degrees")
    plt.ylabel("Frequency")

    plt.subplot(1,2,2)
    plt.hist(trans_errors, bins=30)
    plt.title("Translation Error Histogram")
    plt.xlabel("Degrees")
    plt.ylabel("Frequency")

    plt.tight_layout()

    # save figure
    plt.savefig(save_path, dpi=300)

    print(f"Saved histogram figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()



# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    # add argument parsing for video name and whether to use LoRA or not
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="cskpnp1m39g9b110ri229wyu_10s")
    parser.add_argument("--use_lora", action="store_true")
    args = parser.parse_args()
    use_LoRA = args.use_lora
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vid_name = args.video
    output_dir = os.path.join("rel_pose_results", vid_name, "matches_lora" if use_LoRA else "matches")
    os.makedirs(output_dir, exist_ok=True)
    log_file = open(os.path.join("rel_pose_results", vid_name, "evaluation_results_lora.txt" if use_LoRA else "evaluation_results.txt"), "w")
    sys.stdout = log_file
    print(F"Video: {vid_name}   | Using LoRA: {use_LoRA}")
    pairs_file = f"labeled_videos\\{vid_name}\\pairs_with_pose.txt"
    root_frames = Path(f"labeled_videos\\{vid_name}")

    # load pairs
    K, width, height, pairs = load_pairs_file(pairs_file)

    print("Loaded pairs:",len(pairs))
    print("Camera matrix:\n",K)

    imagenet_mean=[0.485,0.456,0.406]
    imagenet_std=[0.229,0.224,0.225]

    trfs = Compose([
        ToTensor(),
        Normalize(imagenet_mean,imagenet_std)
    ])

    # load CroCo
    ckpt = torch.load(
        "pretrained_models/CroCo_V2_ViTBase_SmallDecoder.pth",
        map_location="cpu"
    )

    model = CroCoNet(**ckpt["croco_kwargs"])
    model.load_state_dict(ckpt["model"], strict=True)

    model = model.to(device).eval()
    if use_LoRA:
        lora_path = r"output\LoRA_on_ENCODER_Decoder_r16_qkv_proj_peft_filtered_data_increased_data\checkpoint-LoRA_on_ENCODER_Decoder_r16_qkv_proj_peft_filtered_data_increased_data-best.pth"
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.to(device)

        model.eval()

    rot_errors = []
    trans_errors = []
    match_counts = []

    patch_size = 16

    for i, (img1,img2,T_gt) in enumerate(pairs):

        path1 = root_frames / img1
        path2 = root_frames / img2

        img1_pil = Image.open(path1).convert("RGB")
        img2_pil = Image.open(path2).convert("RGB")

        w1,h1 = img1_pil.size
        w2,h2 = img2_pil.size

        image1 = trfs(img1_pil.resize((224,224))).unsqueeze(0).to(device)
        image2 = trfs(img2_pil.resize((224,224))).unsqueeze(0).to(device)

        feat1 = extract_features(model,image1)
        feat2 = extract_features(model,image2)

        pts1,pts2 = dense_match(feat1,feat2,patch_size)


        if len(pts1) < 8:
            continue

        # convert back to original resolution
        pts1 = rescale_points(pts1,w1,h1)
        pts2 = rescale_points(pts2,w2,h2)
        img1_name = img1.split("/")[-1][:-4]  # Remove .jpg extension
        img2_name = img2.split("/")[-1][:-4]
        save_path = f"matches_{img1_name}_{img2_name}.jpg"
        save_path = os.path.join(output_dir, save_path)
        save_matches(path1, path2, pts1, pts2, save_path, max_matches=200)
        # exit()
        
        R_pred,t_pred = estimate_pose(pts1,pts2,K)
        match_counts.append(len(pts1))

        if R_pred is None:
            continue

        rot_err,trans_err = pose_error(T_gt,R_pred,t_pred)

        rot_errors.append(rot_err)
        trans_errors.append(trans_err)

        print(
            f"{img1} -> {img2} | "
            f"matches={len(pts1)} | "
            f"rot={rot_err:.2f}° | "
            f"trans={trans_err:.2f}°"
        )

    print("\n===== FINAL RESULTS =====")

    print("Pairs evaluated:",len(rot_errors))

    if len(rot_errors) > 0:

        print_statistics(rot_errors,trans_errors)

        print("\nMatching Statistics")
        print("Avg Matches :", np.mean(match_counts))

        plot_error_histograms(rot_errors, trans_errors, save_path=os.path.join("rel_pose_results", vid_name, "pose_error_histograms.png") if not use_LoRA else os.path.join("rel_pose_results", vid_name, "pose_error_histograms_lora.png") , show=False)


if __name__ == "__main__":
    main()