from torch.utils.data import Dataset
import os

class BratsDataset(Dataset):
    def __init__(self, root: str, fnames: list[str] = None):
        self.root = root
        self.fnames = fnames
        self._build_index()

    def _build_index(self):
        images_path = os.path.join(self.root, "imagesTr")
        labels_path = os.path.join(self.root, "labelsTr")

        self.images_and_labels = []

        for fname in os.listdir(images_path):
            if fname.endswith("nii.gz") and fname.startswith("BRATS"):
                image_path = os.path.join(images_path, fname)
                label_path = os.path.join(labels_path, fname)
                self.images_and_labels.append({"image": image_path, "label": label_path, "image_fname": image_path.split(os.sep)[-1]})

    def __len__(self):
        return len(self.images_and_labels)
    

    def __getitem__(self, index):
        img_label_dict = self.images_and_labels[index]        
        return img_label_dict