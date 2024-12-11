from torchvision import transforms
from .ContrastiveCrop import ContrastiveCrop
from .misc import MultiViewTransform, CCompose, GaussianBlur
from torchvision.transforms import Compose

class Self_Compose(Compose):
    def __call__(self, x):  # x: [sample, box]
        x = x[0]
        for t in self.transforms[0:]:
            x = t(x)
        return x


def imagenet_pretrain_rcrop(mean=None, std=None):
    trans_list = [
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    transform = transforms.Compose(trans_list)
    transform = MultiViewTransform(transform, transform, num_views=2)
    return transform


def imagenet_pretrain_ccrop(alpha=0.6, mean=None, std=None):
    trans_list = [
        ContrastiveCrop(alpha=alpha, size=224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    transform1 = CCompose(trans_list)

    trans_list2 = [
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    transform2 = Self_Compose(trans_list2)

    transform = MultiViewTransform(transform1, transform2, num_views=2)
    return transform


def imagenet_linear_train(mean=None, std=None):
    trans = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return trans


def imagenet_val(mean=None, std=None):
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return trans


def imagenet_eval_boxes(mean=None, std=None):
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return trans


def imagenet_supervised_train(normalize):
    trans = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        normalize,
    ])
    return trans
