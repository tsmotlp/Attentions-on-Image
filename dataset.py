import torch
import torchvision
import torchvision.transforms as transforms

class Train_Data():
    def __init__(self, opts):
        self.opts = opts
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # the label of cifar10
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.train_loader = self.load_train_data()
    def load_train_data(self):
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.opts.batch_size, shuffle=True, num_workers=1)
        return train_loader


class Test_Data():
    def __init__(self, opts):
        self.opts = opts
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # the label of cifar10
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.test_loader = self.load_test_data()
    def load_test_data(self):
        test_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.opts.batch_size, shuffle=False, num_workers=1)
        return test_loader