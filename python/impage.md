## Image 处理

tensorflow/python/ops/image_ops.py

对于读到的图片内容，先解码，之后开始处理，如翻转， 调整对比，调整亮度等等

def decode_image(contents, channels=None, name=None) : 将图片解码为 Tensor

def random_flip_left_right(image, seed=None)

def adjust_contrast(images, contrast_factor)

def adjust_brightness(image, delta) : 调整亮度

