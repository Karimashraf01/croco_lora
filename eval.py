import argparse
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from models.croco import CroCoNet
from models.criterion import MaskedMSE
from datasets.custom_pairs_dataset import DronePairsDataset


def get_args():
    parser = argparse.ArgumentParser("CroCo evaluation")

    parser.add_argument(
        "--pairs_txt",
        default="pairs_office_0024_out.txt",
        # required=True,
        help="Pairs txt file to evaluate"
    )

    parser.add_argument(
        "--data_dir",
        default="old_dataset"
    )

    parser.add_argument(
        "--batch_size",
        default=1,
        type=int
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int
    )

    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    total_samples = 0

    for image1, image2 in loader:

        image1 = image1.to(device)
        image2 = image2.to(device)

        out, mask, target = model(image1, image2)

        loss = criterion(out, mask, target)

        batch_size = image1.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def main():

    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")

    dataset = DronePairsDataset(
        Path(args.data_dir) / args.pairs_txt
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print("Loading checkpoint...")

    # Load base CroCo model
    ckpt = torch.load(
        "pretrained_models/CroCo_V2_ViTBase_SmallDecoder.pth",
        map_location="cpu"
    )

    model = CroCoNet(**ckpt["croco_kwargs"])
    model.load_state_dict(ckpt["model"], strict=True)

    model.to(device)

    criterion = MaskedMSE(norm_pix_loss=True)

    print("Running evaluation...")

    loss = evaluate(model, loader, criterion, device)

    print("=================================")
    print("Evaluation results for", args.pairs_txt, ":")
    print("Average Masked MSE:", loss)
    print("=================================")


if __name__ == "__main__":
    main()