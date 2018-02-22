

### 预处理

def normalize_image(image, original_minval, original_maxval, target_minval,target_maxval)

将  image 的范围从 [original_minval, original_maxval] 变为 [target_minval, target_maxval]

def random_horizontal_flip(image, boxes=None, masks=None, keypoints=None, keypoint_flip_permutation=None, seed=None)

根据 50% 概率，随机将 image, boxes, masks, keypoints
左右翻转，返回元组（包含变化之后的 imges, boxes, masks, keypoints)

def random_rotation90(image, boxes=None, masks=None, keypoints=None, seed=None)

根据 50% 概率，随机将 image, boxes, masks, keypoints
旋转 90 度，返回元组（包含变化之后的 imges, boxes, masks, keypoints)

def random_pixel_value_scale(image, minval=0.9, maxval=1.1, seed=None)

将 image 每一个元素乘以 minval, maxval 之间的随机数，之前确保每个元素最大为 1.0,
最小为 0

def random_image_scale(image, masks=None, min_scale_ratio=0.5, max_scale_ratio=2.0, seed=None)

将  image 的 height 和 width 随机扩展到 [min_scale_ratio, max_scale_ratio]
之间，并将 image 和 masks 的 shape 设置为该值

def random_rgb_to_gray(image, probability=0.1, seed=None)

将 image 元素的范围由 rgb 变为 gray

def random_adjust_brightness(image, max_delta=0.2)

调整 image 的亮度，每个像素取值在  e * [-max_delta, max_delta]

def random_adjust_contrast(image, min_delta=0.8, max_delta=1.25)

调整 image 的对比度，在 [min_delta, max_delta] 之间

def random_adjust_hue(image, max_delta=0.02)

调整 image 的 hue(色彩)，在 [-max_delta, max_delta 之间

def random_adjust_saturation(image, min_delta=0.8, max_delta=1.25)

调整 image 的饱和度，在 [min_delta, max_delta 之间

def random_distort_color(image, color_ordering=0)

对 image 进行扭曲操作，具体根据 color_ordering 来调整。实际是联合 brightness,
stauration, hue, contrast 来进行

def random_jitter_boxes(boxes, ratio=0.05, seed=None)

TODO

def random_crop_image(image, boxes, labels, label_scores=None, masks=None,
    keypoints=None, min_object_covered=1.0, aspect_ratio_range=(0.75, 1.33),
    area_range=(0.1, 1.0), overlap_thresh=0.3, random_coef=0.0, seed=None)

TODO

def random_pad_image(image, boxes,
                     min_image_size=None,
                     max_image_size=None,
                     pad_color=None,
                     seed=None)

1. 把 image 变为 min_image_size, max_image_size 之间的图片 new_image
2. 将增加的部分，随机分成两部分，分别加在 image 的前面和后面，对于增加的部分全部置 0
3. 对于新增加部分，加上 pad_color
4. 设置 box

def random_crop_pad_image(image,
                          boxes,
                          labels,
                          label_scores=None,
                          min_object_covered=1.0,
                          aspect_ratio_range=(0.75, 1.33),
                          area_range=(0.1, 1.0),
                          overlap_thresh=0.3,
                          random_coef=0.0,
                          min_padded_size_ratio=(1.0, 1.0),
                          max_padded_size_ratio=(2.0, 2.0),
                          pad_color=None,
                          seed=None)

1. 调用 random_crop_image 对  image 进行 crop
2. 调用 random_pad_image 对  crop 之后的  image 进行 pad

def random_crop_to_aspect_ratio(image,
                                boxes,
                                labels,
                                label_scores=None,
                                masks=None,
                                keypoints=None,
                                aspect_ratio=1.0,
                                overlap_thresh=0.3,
                                seed=None)

TODO

def random_pad_to_aspect_ratio(image,
                               boxes,
                               masks=None,
                               keypoints=None,
                               aspect_ratio=1.0,
                               min_padded_size_ratio=(1.0, 1.0),
                               max_padded_size_ratio=(2.0, 2.0),
                               seed=None)

TODO

def random_black_patches(image,
                         max_black_patches=10,
                         probability=0.5,
                         size_to_image_ratio=0.1,
                         random_seed=None)

总共执行 max_black_patches 次，每次随机地将 image 部分像素置为 0，。置为 0 的部分为 [x_min, image_width], [y_min, image_height]

y_min = image.height * random(0, 1 - size_to_image_ratio)
x_min = image.width * random(0, 1 - size_to_image_ratio)
box_size = min(image.height, image.width) * size_to_image_ratio
black_box = tf.ones([box_size, box_size, 3])
mask = 1 - tf.image.pad_to_bounding_box(black_box, y_min, x_min,image_height, image_width)
image = image * mask


def random_resize_method(image, target_size)

从 BILINEAR, NEAREST_NEIGHBOR, BICUBIC, AREA 中随机选择一个方法，将  image
的 size 置为 target_size

def preprocess(tensor_dict, preprocess_options, func_arg_map=None)
