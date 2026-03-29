import matplotlib.pyplot as plt
import json
import argparse
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    all_dice_scores = []
    all_edema_dice_scores = []
    all_nt_dice_scores = []
    all_et_dice_scores = []

    for fold in range(5):
        with open(os.path.join(args.output_dir, str(fold), "logs.json"), "r") as f:
            obj = json.load(f)

        all_dice_scores.append(obj["dice"])
        all_edema_dice_scores.append(obj["edema_dice"])
        all_et_dice_scores.append(obj["et_dice"])
        all_nt_dice_scores.append(obj["nt_dice"])

    epochs = [e + 1 for e in range(all_dice_scores[0])]

    average_dice_scores = np.asarray(all_dice_scores).mean(axis=0)
    average_edema_dice_scores = np.asarray(all_edema_dice_scores).mean(axis=0)
    average_et_dice_scores = np.asarray(all_et_dice_scores).mean(axis=0)
    average_nt_dice_scores = np.asarray(all_nt_dice_scores).mean(axis=0)
    
    
    plt.plot(epochs, average_dice_scores, label = "Average Dice (All classes)")
    plt.plot(epochs, average_edema_dice_scores, label = "Average Dice (Edema)")
    plt.plot(epochs, average_nt_dice_scores, label = "Average Dice (NT)")
    plt.plot(epochs, average_et_dice_scores, label = "Average Dice (ET)")
    
    plt.xlabel("Epochs")
    plt.ylabel("Dice")

    plt.legend()

    plt.savefig(os.path.join(args.output_dir, "average_dice.png"))

if __name__ == "__main__":
    main()