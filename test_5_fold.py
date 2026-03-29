from dataset.BratsDataset import BratsDataset
import argparse
import torch
import numpy as np
from monai.data import DataLoader, decollate_batch, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
import os
import json
from medpy.metric import binary
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

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

def combine_mask(mask: torch.Tensor, labels_to_combine):
    result = torch.zeros(mask.shape[1:], device=mask.device, dtype=torch.bool)
    for label in labels_to_combine:
        result |= mask[label].bool()
    return result.unsqueeze(0).float()

def compute_hd(pred: torch.Tensor, gt: torch.Tensor):
    pred = pred.squeeze(0).detach().cpu().numpy().astype(bool)
    gt = gt.squeeze(0).detach().cpu().numpy().astype(bool)

    if pred.any() and gt.any():
        return binary.hd(pred, gt)
    else:
        return np.nan
    
def compute_hd95(pred: torch.Tensor, gt: torch.Tensor):
    pred = pred.squeeze(0).detach().cpu().numpy().astype(bool)
    gt = gt.squeeze(0).detach().cpu().numpy().astype(bool)

    if pred.any() and gt.any():
        return binary.hd95(pred, gt)
    else:
        return np.nan


def main():
    args = get_args()

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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    use_amp = device.type == "cuda"

    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_et = DiceMetric(include_background=True, reduction="mean")
    dice_metric_wt = DiceMetric(include_background=True, reduction="mean")
    dice_metric_tc = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    
    for fold in range(5):
        fold_dir = os.path.join(args.output_dir, str(fold))
        if not os.path.exists(fold_dir):
            raise ValueError(f"Dir: {fold_dir} does not exist. Make sure the output_dir has 5 folds from the training script")
        
        model_path = os.path.join(fold_dir, "best_metric_model.pth")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)

        with open(os.path.join(fold_dir, "logs.json"), "r") as f:
            obj = json.load(f)
            val_fnames = obj["val_images"]

        _ds = BratsDataset(root=args.root, fnames=val_fnames)
        ds = Dataset(_ds, transform=val_transform)
        loader = DataLoader(ds, num_workers=1, batch_size=1, shuffle=False)

        hds_et = []
        hd95s_et = []

        hds_wt = []
        hd95s_wt = []

        hds_tc = []
        hd95s_tc = []

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=use_amp):
                for val_data in tqdm(loader, f"validating fold: {fold + 1}"):
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )

                    val_outputs = sliding_window_inference(inputs=val_inputs, roi_size=(192, 192, 128), sw_batch_size=1, predictor=model, overlap=0.5)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

                    val_labels = decollate_batch(val_labels)

                    et_output_mask = list(map(lambda b: combine_mask(b, [2]), val_outputs))
                    tc_output_mask = list(map(lambda b: combine_mask(b, [1, 2]), val_outputs))
                    wt_output_mask = list(map(lambda b: combine_mask(b, [0, 1, 2]), val_outputs))

                    et_labels_mask = list(map(lambda b: combine_mask(b, [2]), val_labels))
                    tc_labels_mask = list(map(lambda b: combine_mask(b, [1, 2]), val_labels))
                    wt_labels_mask = list(map(lambda b: combine_mask(b, [0, 1, 2]), val_labels))


                    dice_metric_tc(y_pred=tc_output_mask, y=tc_labels_mask)
                    dice_metric_et(y_pred=et_output_mask, y=et_labels_mask)
                    dice_metric_wt(y_pred=wt_output_mask, y=wt_labels_mask)

                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)


                    for b in range(len(val_labels)):
                        hds_et.append(compute_hd(pred=et_output_mask[b], gt=et_labels_mask[b]))
                        hd95s_et.append(compute_hd95(pred=et_output_mask[b], gt=et_labels_mask[b]))

                        hds_wt.append(compute_hd(pred=wt_output_mask[b], gt=wt_labels_mask[b]))
                        hd95s_wt.append(compute_hd95(pred=wt_output_mask[b], gt=wt_labels_mask[b]))

                        hds_tc.append(compute_hd(pred=tc_output_mask[b], gt=tc_labels_mask[b]))
                        hd95s_tc.append(compute_hd95(pred=tc_output_mask[b], gt=tc_labels_mask[b]))


        metric = dice_metric.aggregate().item()
        metric_batch = dice_metric_batch.aggregate()
        metric_edema = metric_batch[0].item()
        metric_nt = metric_batch[1].item()

        metric_wt = dice_metric_wt.aggregate().item()
        metric_tc = dice_metric_tc.aggregate().item()
        metric_et = dice_metric_et.aggregate().item()
        
        
        mean_hd_et = np.nanmean(hds_et)
        mean_hd95_et = np.nanmean(hd95s_et)

        mean_hd_wt = np.nanmean(hds_wt)
        mean_hd95_wt = np.nanmean(hd95s_wt)

        mean_hd_tc = np.nanmean(hds_tc)
        mean_hd95_tc = np.nanmean(hd95s_tc)

        dice_metric.reset()
        dice_metric_batch.reset()
        dice_metric_tc.reset()
        dice_metric_et.reset()
        dice_metric_wt.reset()

        print(
            f"Fold: {fold + 1} | mean dice: {metric:.4f}"
            f"\nenhancing tumor: {metric_et:.4f} whole tumor dice: {metric_wt:.4f} tumor core dice: {metric_tc:.4f} | non-enhancing tumor dice: {metric_nt:.4f} edema dice: {metric_edema:.4f}"
            f"\n enhancing tumor hd: {mean_hd_et:.4f} whole tumor hd: {mean_hd_wt:.4f} tumor core hd: {mean_hd_tc:.4f}"
            f"\n enhancing tumor hd95: {mean_hd95_et:.4f} whole tumor hd95: {mean_hd95_wt:.4f} tumor core hd95: {mean_hd95_tc:.4f}"
        )
        print("---" * 10)

if __name__ == "__main__":
    main()
        

        
