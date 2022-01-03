import torch
from torchvision import transforms, datasets

class data_loader:
    def __init__(
            self, preproc_treatment, batch_size, crop_size=None, resize_size=None, normalize_param=None):
        self.preproc_treatment = preproc_treatment
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.normalize_param = normalize_param

    def define_preprocess(self):
        if self.preproc_treatment == 'yes':
            self.data_transform = transforms.Compose([
                transforms.CenterCrop(self.crop_size),
                transforms.Resize(self.resize_size),
                transforms.ToTensor(),
                transforms.Normalize(self.normalize_param[0], self.normalize_param[1])])
        elif self.preproc_treatment == 'no':
            self.data_transform = transforms.Compose([
                transforms.ToTensor()])

        print(self.data_transform)

    def build_loader(self):
        train_data = datasets.ImageFolder(
            './Trail_dataset/train_data',
             transform = self.data_transform)

        test_data = datasets.ImageFolder(
            './Trail_dataset/test_data',
             transform = self.data_transform)

        train_set_size = int(len(train_data) * 0.8)
        valid_set_size = len(train_data) - train_set_size

        train_data, valid_data = torch.utils.data.random_split(train_data, [train_set_size, valid_set_size])

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=8)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=self.batch_size, shuffle=True, num_workers=8)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

        return train_loader, valid_loader, test_loader

