import os

folder = "old_dataset\\office_0024_out\\content\\nyu_data\\data\\nyu2_train\\office_0024_out"   # folder containing images
step = 10
output_file = "pairs_office_0024_out.txt"

# get all jpg files
images = [f for f in os.listdir(folder) if f.endswith(".jpg")]

# sort numerically (1.jpg, 2.jpg, ...)
images = sorted(images, key=lambda x: int(os.path.splitext(x)[0]))

pairs = []

for i in range(0, len(images) - step, step):
    img1 = images[i]
    img2 = images[i + step]
    pairs.append(f"{folder}/{img1} {folder}/{img2}")

with open(output_file, "w") as f:
    for p in pairs:
        f.write(p + "\n")

print("Saved", len(pairs), "pairs to", output_file)