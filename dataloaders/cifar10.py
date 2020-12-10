import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def cifar10_dataloaders(train_batch_size=64, test_batch_size=100, num_workers=2, data_dir = 'datasets/cifar10'):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    # train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    train_loader = Batches(train_set, batch_size=train_batch_size, shuffle=True, set_random_choices=False, num_workers=0, drop_last=True)
    
    # train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    # val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    # val_loader = DataLoader(val_set, batch_size=test_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    

    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)
    # test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = Batches(train_set, batch_size=test_batch_size, shuffle=True, set_random_choices=False, num_workers=0, drop_last=True)
    
    # return train_loader, val_loader, test_loader
    return train_loader, test_loader


def get_adversarial_images(adversarial_data="autoattack", batch_size=64) :

    train_adv_images = None
    train_adv_labels = None
    test_robust_images = None
    test_robust_labels = None

    adv_dir = "adv_examples/{}/".format(adversarial_data)
    train_path = adv_dir + "train.pth" 
    test_path = adv_dir + "test.pth"
    
    if adversarial_data in ["autoattack", "autopgd", "bim", "cw", "deepfool", "fgsm", "pgd", "newtonfool", "pixelattack", "spatialtransformation", "squareattack"] :
        adv_train_data = torch.load(train_path)
        train_adv_images = adv_train_data["adv"]
        train_adv_labels = adv_train_data["label"]
        adv_test_data = torch.load(test_path)
        test_robust_images = adv_test_data["adv"]
        test_robust_labels = adv_test_data["label"]        
    elif adversarial_data in ["ffgsm", "mifgsm", "tpgd"] :
        adv_data = {}
        adv_data["adv"], adv_data["label"] = torch.load(train_path)
        train_adv_images = adv_data["adv"].numpy()
        train_adv_labels = adv_data["label"].numpy()
        adv_data = {}
        adv_data["adv"], adv_data["label"] = torch.load(test_path)
        test_robust_images = adv_data["adv"].numpy()
        test_robust_labels = adv_data["label"].numpy()
    else :
        raise ValueError("Unknown adversarial data")
        
#     print("")
#     print("Train Adv Attack Data: ", adversarial_data)
#     print("Dataset shape: ", train_adv_images.shape)
#     print("Dataset type: ", type(train_adv_images))
#     print("Label shape: ", len(train_adv_labels))
#     print("")
    
    train_adv_set = list(zip(train_adv_images,
        train_adv_labels))
    
    train_adv_batches = Batches(train_adv_set, batch_size, shuffle=True, set_random_choices=False, num_workers=0, drop_last=True)
    
    test_robust_set = list(zip(test_robust_images,
        test_robust_labels))
        
    test_robust_batches = Batches(test_robust_set, batch_size, shuffle=True, num_workers=0, drop_last=True)

    return train_adv_batches, test_robust_batches
