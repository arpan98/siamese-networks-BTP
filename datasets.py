import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

TRAIN_TEST_SPLIT = 0.8

class OlivettiDataset(Dataset):

    def __init__(self, olivetti_dataset, train, transform):
        self.train = train
        self.transform = transform
        self.olivetti_dataset = olivetti_dataset
        images_per_subject = 10
        num_subjects = 40
        self.train_data = olivetti_dataset.images[:int(TRAIN_TEST_SPLIT * images_per_subject * num_subjects),:,:]
        self.train_labels = olivetti_dataset.target[:int(TRAIN_TEST_SPLIT * images_per_subject * num_subjects)]
        self.test_data = olivetti_dataset.images[int(TRAIN_TEST_SPLIT * images_per_subject * num_subjects) + 1:,:,:]
        self.test_labels = olivetti_dataset.target[int(TRAIN_TEST_SPLIT * images_per_subject * num_subjects) + 1:]

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index]
            siamese_index = index
            if target == 1:
                while True:
                    siamese_index = np.random.randint(len(self.train_labels))
                    if self.train_labels[siamese_index] == label1:
                        break
            else:
                while True:
                    siamese_index = np.random.randint(len(self.train_labels))
                    if self.train_labels[siamese_index] != label1:
                        break
            img2, label2 = self.train_data[siamese_index], self.train_labels[siamese_index]
        else:
            img1 = self.test_data[index]
            label1 = self.test_labels[index]
            index2 = np.random.randint(0, self.test_data.shape[0])
            img2 = self.test_data[index2]
            label2 = self.test_labels[index2]
            target = int(label1 == label2)
        img1 = 255 * img1
        img1 = np.stack((img1.astype(np.uint8),)*3, -1)
        img1 = Image.fromarray(img1, mode='RGB')
        img2 = 255 * img2
        img2 = np.stack((img2.astype(np.uint8),)*3, -1)
        img2 = Image.fromarray(img2, mode='RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, target

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)
