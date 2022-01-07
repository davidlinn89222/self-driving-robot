import torch
from torchvision import transforms, datasets

class data_loader:
    def __init__(
            self, preproc_treatment, batch_size, preprocess_para):
        self.preproc_treatment = preproc_treatment
        self.batch_size = batch_size
        self.preprocess_para = preprocess_para

    def define_preprocess(self):
        if self.preproc_treatment == 'yes':
            trans_lst = []
            for key in self.preprocess_para.keys():
                if key == 'rd_crop_size':
                    trans_lst.append(transforms.RandomCrop(self.preprocess_para["rd_crop_size"]))
                elif key == 'resize_size':
                    trans_lst.append(transforms.Resize(self.preprocess_para['resize_size']))
                elif key == 'normalize_param':
                    trans_lst.append(transforms.Normalize(self.preprocess_para['normalize_param'][0], self.preprocess_para['normalize_param'][1]))
                elif key == 'brightness':
                    trans_lst.append(transforms.ColorJitter(brightness = self.preprocess_para['brightness'], contrast=0, hue=0))
                elif key == 'totensor':
                    trans_lst.append(transforms.ToTensor())
                elif key == 'dncrop':
                    trans_lst.append(transforms.Lambda(lambda x: transforms.functional.crop(x, self.preprocess_para['dncrop']["top"], self.preprocess_para['dncrop']["left"], self.preprocess_para['dncrop']["height"], self.preprocess_para['dncrop']["width"])))
            self.data_transform = transforms.Compose(trans_lst)
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

        train_data, valid_data = torch.utils.data.random_split(train_data, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(123))

        if self.preproc_treatment == 'yes':
            if 'num_of_worker' in self.preprocess_para.keys():
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.preprocess_para['num_of_worker'])
                valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=self.batch_size, shuffle=True, num_workers=self.preprocess_para['num_of_worker'])
        else
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=16)
            valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=self.batch_size, shuffle=True, num_workers=16)

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

        return train_loader, valid_loader, test_loader

