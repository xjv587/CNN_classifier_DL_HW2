from PIL import Image
import os
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    """
    WARNING: Do not perform data normalization here. 
    """
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use your solution (or the master solution) to HW1
        """
        self.dataset_path = dataset_path
        self.labels_file = os.path.join(dataset_path, 'labels.csv')
        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.class_mapping = {'background': 0, 'kart': 1, 'pickup': 2, 'nitro': 3, 'bomb': 4, 'projectile': 5}
        self.data = self._load_data()

    def _load_data(self):
        data = []
        with open(self.labels_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_path = os.path.join(self.dataset_path, row['file'])
                label = int(self.class_mapping[row['label']])
                data.append((image_path, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        return a tuple: img, label
        """
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        label = label
        image = self.transforms(image)
        return image, label


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
