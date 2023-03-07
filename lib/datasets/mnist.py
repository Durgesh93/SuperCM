import numpy as np
from PIL import Image
import os

class MNIST:

    def __init__(self, root, split="l_train"):
        self.dataset = np.load(os.path.join(root, "mnist", split+".npy"), allow_pickle=True).item()

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        h,w   = image.shape
        image = image.reshape(1,h,w)
        image = np.repeat(image,3,axis=0)
        label = self.dataset["labels"][idx]
        image = (image/255. - 0.5)/0.5

        if "labels2" in self.dataset:
            label2 = self.dataset["labels2"][idx]
            return image, label, label2
        else:
            return image, label


    def __len__(self):
        return len(self.dataset["images"])