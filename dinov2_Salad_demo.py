import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# -----------------------
# Load model from TorchHub
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.hub.load("serizba/salad", "dinov2_salad", trust_repo=True)
model = model.to(device)
model.eval()

# -----------------------
# Image preprocessing
# -----------------------
transform = T.Compose([
    T.Resize((322, 322)),   # size used in SALAD repo
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img)
    return img.unsqueeze(0).to(device)



if __name__ == "__main__":
    pairs_txt = "New_drone_dataset/croco_pairs_train.txt"
    counter = 0
    with open(pairs_txt, "r") as f:
        pairs = [line.strip().split() for line in f.readlines()]
    for im1_path, im2_path in pairs:
        img1 = load_image(im1_path)
        img2 = load_image(im2_path)

        with torch.inference_mode():
            desc1 = model(img1)
            desc2 = model(img2)

        desc1 = F.normalize(desc1, dim=1)
        desc2 = F.normalize(desc2, dim=1)

        sim = torch.mm(desc1, desc2.t()).item()
        if (sim+1)/2 > 0.8:
            counter += 1
        
        print(f"Similarity between {im1_path} and {im2_path}: {(sim+1)/2:.4f}")
    print(f"Number of similar pairs: {counter}")
