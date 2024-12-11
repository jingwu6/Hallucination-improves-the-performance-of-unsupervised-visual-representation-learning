from torchvision.transforms import Compose
from PIL import ImageFilter
import random
from torchvision import transforms

class CCompose(Compose):
    def __call__(self, x):  # x: [sample, box]
        img = self.transforms[0](*x)
        for t in self.transforms[1:]:
            img = t(img)
        return img


class MultiViewTransform:
    """Create multiple views of the same image"""

    def __init__(self, transform1, transform2, num_views=2):
        if not isinstance(transform1, (list, tuple)):
            # transform = [transform for _ in range(num_views)]
            transform = [transform1, transform2]
        self.transforms = transform

    def __call__(self, x):
        # views = [t(x) for t in self.transforms]
        views = []
        for i in range(len(self.transforms)):
            t = self.transforms[i]
            views.append(t(x))
        return views


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
