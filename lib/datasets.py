from torchvision.datasets import CIFAR10, CIFAR100, SVHN


class CIFAR10withIndex(CIFAR10):
    def __getitem__(self, index):
        img, target = super(CIFAR10withIndex, self).__getitem__(index)

        return img, target, index


class CIFAR100withIndex(CIFAR100):
    def __getitem__(self, index):
        img, target = super(CIFAR100withIndex, self).__getitem__(index)
        return img, target, index


class SVHNwithIndex(SVHN):
    def __getitem__(self, index):
        retval = super(SVHN, self).__getitem__(index)
        return retval + (index,)
