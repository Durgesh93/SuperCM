import numpy as np
import os

class SVHN:
    def __init__(self, root, split="l_train"):
        self.dataset = np.load(os.path.join(root, "svhn", split+".npy"), allow_pickle=True).item()

    def __getitem__(self, idx):
        
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        image = (image/255. - 0.5)/0.5

        if "labels2" in self.dataset:
            label2 = self.dataset["labels2"][idx]
            return image, label, label2
        else:
            return image, label


    def __len__(self):
        return len(self.dataset["images"])