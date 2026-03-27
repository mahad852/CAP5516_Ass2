from dataset.BratsDataset import BratsDataset
import argparse
import torch
import numpy as np
from monai.data import DataLoader, decollate_batch, Dataset, CacheDataset
from torch.utils.data import random_split
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import AttentionUnet, SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
import os
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=1)

    args = parser.parse_args()
    return args

class Debug(MapTransform):
    def __call__(self, data):
        print(data.keys())
        print(data["image"].shape, "image shape")
        print(data["label"].shape, "label shape")
        

def main():
    args = get_args()

    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Debug(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            Debug(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    ds = BratsDataset(root=args.root)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for fold in range(5):
        fold_dir = os.path.join(args.output_dir, str(fold))
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

        fold_logs = os.path.join(fold_dir, "logs.jsonl")
        with open(fold_logs, "w") as f:
            pass

        _train_ds, _val_ds = random_split(dataset=ds, lengths=[0.8, 0.2])

        train_ds = CacheDataset(_train_ds, transform=train_transform)
        val_ds = CacheDataset(_val_ds, transform=val_transform)

        train_loader = DataLoader(train_ds, num_workers=1, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, num_workers=1, batch_size=8, shuffle=False)

        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
        ).to(device)
        loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
        optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        dice_metric = DiceMetric(include_background=True, reduction="mean")
        dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        best_metric = -1
        best_metric_epoch = -1
        best_metrics_epochs_and_time = [[], [], []]
        epoch_loss_values = []
        metric_values = []
        metric_values_tc = []
        metric_values_wt = []
        metric_values_et = []

        for epoch in range(args.epochs):
            print(f"epoch {epoch + 1}/{args.epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
                optimizer.zero_grad()
                with torch.autocast("cuda"):
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                print(
                    f"{step}/{len(train_ds) // train_loader.batch_size}"
                    f", train_loss: {loss.item():.4f}"
                )
            lr_scheduler.step()
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % args.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )
                        val_outputs = sliding_window_inference(inputs=val_inputs, roi_size=(240, 240, 160), sw_batch_size=1, predictor=model, overlap=0.5)
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                        dice_metric(y_pred=val_outputs, y=val_labels)
                        dice_metric_batch(y_pred=val_outputs, y=val_labels)

                    metric = dice_metric.aggregate().item()
                    metric_values.append(metric)
                    metric_batch = dice_metric_batch.aggregate()
                    metric_tc = metric_batch[0].item()
                    metric_values_tc.append(metric_tc)
                    metric_wt = metric_batch[1].item()
                    metric_values_wt.append(metric_wt)
                    metric_et = metric_batch[2].item()
                    metric_values_et.append(metric_et)
                    dice_metric.reset()
                    dice_metric_batch.reset()

                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        best_metrics_epochs_and_time[0].append(best_metric)
                        best_metrics_epochs_and_time[1].append(best_metric_epoch)
                        torch.save(
                            model.state_dict(),
                            os.path.join(fold_dir, "best_metric_model.pth"),
                        )
                        print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                        f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                        f"\nbest mean dice: {best_metric:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )


if __name__ == "__main__":
    main()
