from dataset.BratsDataset import BratsDataset
import argparse
import os
import json
import numpy as np
import torch

from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--modality", type=int, default=0)
    return parser.parse_args()


class ConvertToMultiChanneld(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = torch.zeros((3, *d[key].shape), dtype=torch.float32)
            for c in range(1, 4):
                result[c - 1] = (d[key] == c)
            d[key] = result
        return d

def normalize_image_for_display(img):
    img = np.asarray(img, dtype=np.float32)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img


def mip_along_plane(volume, plane):
    if plane == "axial":
        return np.max(volume, axis=2)
    elif plane == "coronal":
        return np.max(volume, axis=1)
    elif plane == "sagittal":
        return np.max(volume, axis=0)
    else:
        raise ValueError(f"Unknown plane: {plane}")


def maybe_rotate_for_display(img2d):
    return np.rot90(img2d)


def make_overlay(base_slice, edema_slice, nt_slice, et_slice, alpha=0.35):
    base = normalize_image_for_display(base_slice)
    rgb = np.stack([base, base, base], axis=-1)

    color_ed = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    color_nt = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    color_et = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    for mask, color in [
        (edema_slice, color_ed),
        (nt_slice, color_nt),
        (et_slice, color_et),
    ]:
        mask = mask.astype(bool)
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color

    return np.clip(rgb, 0.0, 1.0)


def save_case_figure(image, gt, pred, save_path, title="", modality=0):
    image_np = image[modality].numpy()

    gt_ed = gt[0].numpy().astype(bool)
    gt_nt = gt[1].numpy().astype(bool)
    gt_et = gt[2].numpy().astype(bool)

    pred_ed = pred[0].numpy().astype(bool)
    pred_nt = pred[1].numpy().astype(bool)
    pred_et = pred[2].numpy().astype(bool)

    planes = ["axial", "coronal", "sagittal"]

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    fig.suptitle(title, fontsize=13)

    for row, plane in enumerate(planes):
        base_mip = mip_along_plane(image_np, plane)

        gt_ed_mip = mip_along_plane(gt_ed.astype(np.uint8), plane).astype(bool)
        gt_nt_mip = mip_along_plane(gt_nt.astype(np.uint8), plane).astype(bool)
        gt_et_mip = mip_along_plane(gt_et.astype(np.uint8), plane).astype(bool)

        pred_ed_mip = mip_along_plane(pred_ed.astype(np.uint8), plane).astype(bool)
        pred_nt_mip = mip_along_plane(pred_nt.astype(np.uint8), plane).astype(bool)
        pred_et_mip = mip_along_plane(pred_et.astype(np.uint8), plane).astype(bool)

        gt_overlay = make_overlay(base_mip, gt_ed_mip, gt_nt_mip, gt_et_mip)
        pred_overlay = make_overlay(base_mip, pred_ed_mip, pred_nt_mip, pred_et_mip)

        gt_overlay = maybe_rotate_for_display(gt_overlay)
        pred_overlay = maybe_rotate_for_display(pred_overlay)

        axes[row, 0].imshow(gt_overlay)
        axes[row, 0].set_title(f"{plane.capitalize()} - Ground Truth")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(pred_overlay)
        axes[row, 1].set_title(f"{plane.capitalize()} - Prediction")
        axes[row, 1].axis("off")

    legend_handles = [
        Patch(facecolor=(1.0, 1.0, 0.0), edgecolor="black", label="Edema"),
        Patch(facecolor=(0.0, 1.0, 1.0), edgecolor="black", label="Non-enhancing tumor"),
        Patch(facecolor=(1.0, 0.0, 0.0), edgecolor="black", label="Enhancing tumor"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_dir = os.path.join(args.output_dir, str(args.fold))
    if not os.path.isdir(fold_dir):
        raise ValueError(f"Fold dir does not exist: {fold_dir}")

    logs_json = os.path.join(fold_dir, "logs.json")
    logs_jsonl = os.path.join(fold_dir, "logs.jsonl")

    if os.path.exists(logs_json):
        logs_path = logs_json
    elif os.path.exists(logs_jsonl):
        logs_path = logs_jsonl
    else:
        raise ValueError(f"Could not find logs.json or logs.jsonl in {fold_dir}")

    with open(logs_path, "r") as f:
        obj = json.load(f)

    val_fnames = obj["val_images"]

    base_ds = BratsDataset(root=args.root, fnames=val_fnames)
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChanneld(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    ds = Dataset(base_ds, transform=val_transform)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
    )

    model_path = os.path.join(fold_dir, "best_metric_model.pth")
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    images_dir = os.path.join(fold_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for idx, batch in tqdm(enumerate(loader), desc="Saving examples..."):
        val_inputs, val_labels = (
            batch["image"].to(device),
            batch["label"].to(device),
        )

        val_outputs = sliding_window_inference(inputs=val_inputs, roi_size=(192, 192, 128), sw_batch_size=1, predictor=model, overlap=0.5)
        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

        pred = val_outputs[0].detach().cpu()
        image = decollate_batch(val_inputs)[0].detach().cpu()
        label = decollate_batch(val_labels)[0].detach().cpu()

        case_name = batch["image_fname"][0]
        save_path = os.path.join(images_dir, f"{case_name}_mip_overlay.png")

        title = f"Fold {args.fold} | {case_name}"
        save_case_figure(
            image=image,
            gt=label,
            pred=pred,
            save_path=save_path,
            title=title,
            modality=args.modality,
        )

        print(f"[{idx + 1}/{len(ds)}] saved: {save_path}")


if __name__ == "__main__":
    main()