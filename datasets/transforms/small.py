from torchvision import transforms
from .ContrastiveCrop import ContrastiveCrop
from .misc import MultiViewTransform, CCompose
from torchvision.transforms import Compose

def cifar_train_rcrop(mean=None, std=None):
    trans_list = [
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    transform = transforms.Compose(trans_list)

    trans_list2 = [
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    transform2 = transforms.Compose(trans_list2)

    transform = MultiViewTransform(transform, transform2, num_views=2)
    return transform


def cifar_train_ccrop(alpha=0.6, mean=None, std=None):
    trans_list = [
        ContrastiveCrop(alpha=alpha, size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    transform = CCompose(trans_list)

    trans_list2 = [
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    transform2 = Self_Compose(trans_list2)
    # transform2 = transforms.Compose(trans_list2)
    
    transform = MultiViewTransform(transform, transform2, num_views=2)
    return transform

class Self_Compose(Compose):
    def __call__(self, x):  # x: [sample, box]
        x = x[0]
        for t in self.transforms[0:]:
            x = t(x)
        return x

def stl10_train_rcrop(mean=None, std=None):
    trans_list = [
        transforms.RandomResizedCrop(size=96, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    transform = transforms.Compose(trans_list)
    transform = MultiViewTransform(transform, transform, num_views=2)
    return transform


def stl10_train_ccrop(alpha=0.6, mean=None, std=None):
    trans_list = [
        ContrastiveCrop(alpha=alpha, size=96, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    transform1 = CCompose(trans_list)

    trans_list2 = [
        transforms.RandomResizedCrop(size=96, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    # transform2 = transforms.Compose(trans_list2)
    transform2 = Self_Compose(trans_list2)

    transform = MultiViewTransform(transform1, transform2, num_views=2)
    return transform


def tiny200_train_rcrop(mean=None, std=None):
    trans_list1 = [
        transforms.RandomResizedCrop(size=64, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    transform1 = transforms.Compose(trans_list1)
    transform = MultiViewTransform(transform1, transform1,num_views=2)
    return transform


def tiny200_train_ccrop(alpha=0.6, mean=None, std=None):
    trans_list1 = [
        ContrastiveCrop(alpha=alpha, size=64, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    transform1 = CCompose(trans_list1)

    trans_list2 = [
        transforms.RandomResizedCrop(size=64, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    transform2 = Self_Compose(trans_list2)

    transform = MultiViewTransform(transform1, transform2, num_views=2)
    return transform


def cifar_linear(mean, std):
    trans = transforms.Compose([
        transforms.RandomResizedCrop(size=32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return trans


def stl10_linear(mean, std):
    trans = transforms.Compose([
        transforms.RandomResizedCrop(size=96),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return trans


def tiny200_linear(mean, std):
    trans = transforms.Compose([
        transforms.RandomResizedCrop(size=64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return trans


def cifar_test(mean, std):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform


def stl10_test(mean, std):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform


def tiny200_test(mean, std):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform
