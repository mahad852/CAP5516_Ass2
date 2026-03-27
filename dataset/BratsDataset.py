from torch.utils.data import Dataset
import os

class BratsDataset(Dataset):
    def __init__(self, root: str):
        self.root = root
        self._build_index()
        
    def _build_index(self):
        images_path = os.path.join(self.root, "imagesTr")
        labels_path = os.path.join(self.root, "labelsTr")

        self.images_and_labels = []

        for fname in os.listdir(images_path):
            image_path = os.path.join(images_path, fname)
            label_path = os.path.join(labels_path, fname)
            self.images_and_labels.append({"image": image_path, "label": label_path})

    def __len__(self):
        return len(self.images_and_labels)
    

    def __getitem__(self, index):
        img_label_dict = self.images_and_labels[index]        
        return img_label_dict