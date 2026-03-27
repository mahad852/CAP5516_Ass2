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
from medpy.metric import binary

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=1)

    args = parser.parse_args()
    return args

class ConvertToMultiChanneld(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = torch.zeros(size=(3, *d[key].size()), dtype=torch.float)
            for c in range(1, 4):    
                result[c - 1] = d[key] == c
            d[key] = result
        return d            

def main():
    args = get_args()

    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
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
    ds = BratsDataset(root=args.root)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for fold in range(5):
        fold_dir = os.path.join(args.output_dir, str(fold))
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

        fold_logs = os.path.join(fold_dir, "logs.json")

        _train_ds, _val_ds = random_split(dataset=ds, lengths=[0.8, 0.2])
        train_images = [d["image"].split(os.sep)[-1] for d in _train_ds]
        val_images = [d["image"].split(os.sep)[-1] for d in _val_ds]

        train_ds = Dataset(_train_ds, transform=train_transform)
        val_ds = Dataset(_val_ds, transform=val_transform)

        train_loader = DataLoader(train_ds, num_workers=1, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_ds, num_workers=1, batch_size=1, shuffle=False)

        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
        ).to(device)

        scaler = torch.amp.GradScaler("cuda")

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
        metric_values_edema = []
        metric_values_nt = []
        metric_values_et = []
        mean_hds = []
        mean_hd95s = []

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
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

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
                batch_hds = []
                batch_hd95s = []

                with torch.no_grad():
                    with torch.autocast(device_type="cuda"):
                        for val_data in val_loader:
                            val_inputs, val_labels = (
                                val_data["image"].to(device),
                                val_data["label"].to(device),
                            )
                            val_outputs = sliding_window_inference(inputs=val_inputs, roi_size=(192, 192, 128), sw_batch_size=1, predictor=model, overlap=0.5)
                            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                            dice_metric(y_pred=val_outputs, y=val_labels)
                            dice_metric_batch(y_pred=val_outputs, y=val_labels)

                            et_labels = val_labels[:, -1].detach().cpu().numpy()
                            et_outputs = torch.stack(val_outputs)[:, -1].detach().cpu().numpy()

                            # for pred, gt in zip(et_outputs, et_labels):
                            #     batch_hds.append(binary.hd(result=pred, reference=gt))
                            #     batch_hd95s.append(binary.hd95(result=pred, reference=gt))

                    metric = dice_metric.aggregate().item()
                    metric_values.append(metric)
                    metric_batch = dice_metric_batch.aggregate()
                    metric_edema = metric_batch[0].item()
                    metric_values_edema.append(metric_edema)
                    metric_nt = metric_batch[1].item()
                    metric_values_nt.append(metric_nt)
                    metric_et = metric_batch[2].item()
                    metric_values_et.append(metric_et)
                    dice_metric.reset()
                    dice_metric_batch.reset()
                    # mean_hds.append(np.asarray(batch_hds).mean())
                    # mean_hd95s.append(np.asarray(batch_hd95s).mean())

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
                        f" edema: {metric_edema:.4f} non-enhancing tunmor: {metric_nt:.4f} enhancing: {metric_et:.4f}"
                        f"\nbest mean dice: {best_metric:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )

        with open(fold_logs, "w") as f:
            json.dump({
                "best_mean_dice" : best_metric, "best_epoch": best_metric_epoch,
                "edema_dice" : metric_values_edema, "nt_dice": metric_values_nt, "et_dice": metric_values_et, "dice": metric_values,
                "train_images": train_images, "val_images": val_images,
            }, f)


if __name__ == "__main__":
    main()
