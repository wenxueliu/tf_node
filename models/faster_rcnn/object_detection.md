



##  预备知识

"SSD: Single Shot MultiBox Detector" Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.


## 模型构建

1. 创建模型
2. 从数据集中读数据
3. 用模型对数据依次做 preprocess, provide_groundtruth, predict, loss 操作
4. 初始化从 checkpoint_path 读取已经训练参数
TODO

### 配置

utils/config_util.py
protos/

1. 在配置文件以字典形式定义配置
2. 创建对应的 proto 文件，并生成对应的  protobuffer 类
3. 通过 tf.gfile.GFile 读取配置文件，用  google.protobuf.text_format 将字符串转为对应的  protobuffer 类对象

def get_configs_from_pipeline_file(pipeline_config_path)

1. 从配置文件 pipeline_config_path 读取配置信息，解析为 pipeline_pb2.TrainEvalPipelineConfig 格式  pipeline_config
2. 用 pipeline_config  各个属性初始化配置字典 config，返回  config

参考 samples/configs

def create_pipeline_proto_from_configs(configs)

将 config 转为 pipeline_pb2.TrainEvalPipelineConfig 格式  pipeline_config

def get_configs_from_multiple_files(model_config_path="",
                                    train_config_path="",
                                    train_input_config_path="",
                                    eval_config_path="",
                                    eval_input_config_path="")

依次读 model_config_path, train_config_path, train_input_config_path,
eval_config_path, eval_input_config_path 初始化配置字典 config

### 预处理

builders.preprocessor_builder.build 复用性很好


image: [batch, height, width, channel] 或 [height, width, channel]
mask: [num_instances, height, width]
box: [num_instances, 4] 其中 4 包含  y_min, x_min, y_max, x_max。num_instances 为 box 的个数



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

def resize_to_range(image,
                    masks=None,
                    min_dimension=None,
                    max_dimension=None,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=False)

large_scale_factor = min_dimension / min(image.width, image.height)
large_size = [image.height * large_scale_factor, image.width * large_scale_factor]
small_scale_factor = max_dimension / max(image.width, image.height)
small_size = [image.height * small_scale_factor, image.width * small_scale_factor]

if max(large_size) > max_dimension return small_size else return large_size //这里还是不理解用意何在

def resize_to_min_dimension(image, masks=None, min_dimension=600)

target_ratio = max(min(image.height, image.width), min_dimension) / min_dimension
target_size = [image.height * target_ratio, image.width * target_ratio]_

用 bilinear 方法将 image 扩展为 target_size。如果 mask 不为空，同样扩展  mask

def scale_boxes_to_pixel_coordinates(image, boxes, keypoints=None)

将  boxes  中的 x_{min|max} 乘以 image.width, y_{min|max} 乘以 image.height
之后返回新的 boxlist

def resize_image(image,
                 masks=None,
                 new_height=600,
                 new_width=1024,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False)

1. 将 image resize 为 [new_height, new_width]
2. 如果 mask 中元素个数多于 0 个， 将 mask resize 为 [new_height, new_width]，否则  reshape 为 [0, new_height, new_width]

def subtract_channel_mean(image, means=None)

image 的每个  channel 的每个像素减去 means

def one_hot_encoding(labels, num_classes=None)

参见下面的例子

a = [0, 1, 2, 2, 1, 0]
one_hot_encoding(a, 3, 1, 0, 0) //[1, 1, 1, 1, 1, 1]

def rgb_to_gray(image)

把 images 的每个 channel 对应的值加起来乘以[0.2989, 0.5870, 0.1140] (此时 shape 为 [height, width])，之后转换为 [height, width, 1]

参考 https://en.wikipedia.org/wiki/Luma_%28video%29


def ssd_random_crop(image,
                    boxes,
                    labels,
                    label_scores=None,
                    masks=None,
                    keypoints=None,
                    min_object_covered=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
                    aspect_ratio_range=((0.5, 2.0),) * 7,
                    area_range=((0.1, 1.0),) * 7,
                    overlap_thresh=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
                    random_coef=(0.15,) * 7,
                    seed=None)

参考 random_crop_image

Liu et al., SSD: Single shot multibox detector.


def ssd_random_crop_pad(image,
                        boxes,
                        labels,
                        label_scores=None,
                        min_object_covered=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
                        aspect_ratio_range=((0.5, 2.0),) * 6,
                        area_range=((0.1, 1.0),) * 6,
                        overlap_thresh=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
                        random_coef=(0.15,) * 6,
                        min_padded_size_ratio=((1.0, 1.0),) * 6,
                        max_padded_size_ratio=((2.0, 2.0),) * 6,
                        pad_color=(None,) * 6,
                        seed=None)

参考 random_crop_pad_image

def ssd_random_crop_fixed_aspect_ratio(
    image,
    boxes,
    labels,
    label_scores=None,
    masks=None,
    keypoints=None,
    min_object_covered=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    aspect_ratio=1.0,
    area_range=((0.1, 1.0),) * 7,
    overlap_thresh=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    random_coef=(0.15,) * 7,
    seed=None)

1. ssd_random_crop
2. random_crop_to_aspect_ratio

def ssd_random_crop_pad_fixed_aspect_ratio(
    image,
    boxes,
    labels,
    label_scores=None,
    masks=None,
    keypoints=None,
    min_object_covered=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    aspect_ratio=1.0,
    aspect_ratio_range=((0.5, 2.0),) * 7,
    area_range=((0.1, 1.0),) * 7,
    overlap_thresh=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    random_coef=(0.15,) * 7,
    min_padded_size_ratio=(1.0, 1.0),
    max_padded_size_ratio=(2.0, 2.0),
    seed=None)

1. ssd_random_crop
2. random_pad_to_aspect_ratio

def preprocess(tensor_dict, preprocess_options, func_arg_map=None)

TODO


### 工具类

def get_variables_available_in_checkpoint(variables, checkpoint_path)

遍历 variables  中的每个变量，返回在 checkpoint_path 中的 get_variable_to_shape_map() 中的变量列表

def filter_variables(variables, filter_regex_list, invert=False)

如果 invert 为  False, 返回 variables 中与  filter_regex_list 都不匹配的变量
如果 invert 为  True, 返回 variables 中与  filter_regex_list 任意匹配的变量

def multiply_gradients_matching_regex(grads_and_vars, regex_list, multiplier)

grads_and_vars 中任意与 regex_list 匹配的变量，乘以 multiplier

def freeze_gradients_matching_regex(grads_and_vars, regex_list)

返回  grads_and_vars 中与 regex_list 不匹配的变量

def merge_boxes_with_multiple_labels(boxes, classes, num_classes)

将 boxes, classes, num_classes  中属于相同  box 的元素进行合并

### input reader



### optimizer

通过配置将 rms, momentum, adam 进行封装，具体参考 optimizer_builder.build

支持 learning_rate 方式

1. 常量
2. exponential_decay
3. 手动设置
4. cosine_decay

注：设计很合理，复用性很好

## model

### faster_rcnn

protos/faster_rcnn.proto

1. preprocessor.resize_image 对输入图像进行处理，具体参考 image_resizer
2. feature_extractor，具体参考 feature_extractor
3. anchor_generator, 具体参考 anchor_generator

rfcn_meta_arch.RFCNMetaArch 或 faster_rcnn_meta_arch.FasterRCNNMetaArch


#### image_resizer

builders/image_resizer_builder.py
builders/preprocessor_builder.py


keep_aspect_ratio_resizer
* BILINEAR
* NEAREST_NEIGHBOR
* BICUBIC
* AREA

fixed_shape_resizer
* BILINEAR
* NEAREST_NEIGHBOR
* BICUBIC
* AREA

#### feature_extractor

protos/faster_rcnn.proto

FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP = {
    'faster_rcnn_nas': frcnn_nas.FasterRCNNNASFeatureExtractor,
    'faster_rcnn_inception_resnet_v2': frcnn_inc_res.FasterRCNNInceptionResnetV2FeatureExtractor,
    'faster_rcnn_inception_v2': frcnn_inc_v2.FasterRCNNInceptionV2FeatureExtractor,
    'faster_rcnn_resnet50': frcnn_resnet_v1.FasterRCNNResnet50FeatureExtractor,
    'faster_rcnn_resnet101': frcnn_resnet_v1.FasterRCNNResnet101FeatureExtractor,
    'faster_rcnn_resnet152': frcnn_resnet_v1.FasterRCNNResnet152FeatureExtractor,
}

frcnn_config.feature_extractor 中保存了 FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP 中值的参数

#### anchor_generator

builders/anchor_generator_builder.py
protos/anchor_generator.proto
protos/grid_anchor_generator.proto
protos/ssd_anchor_generator.proto
anchor_generators/grid_anchor_generator.py
anchor_generators/multiple_grid_anchor_generator.py

* grid_anchor_generator
* ssd_anchor_generator

### hyperparams

protos/hyperparams.proto
builders/hyperparams_builder.py

OP
* slim.fully_connected
* slim.conv2d

regularizer
* slim.l1_regularizer
* slim.l2_regularizer

initializer
* truncated_normal_initializer
* variance_scaling_initializer

activation
* none
* tf.nn.relu
* tf.nn.relu6

### second_stage_box_predictor

protos/box_predictor.proto
core/box_predictor.py
builders/box_predictor_builder.py

* convolutional_box_predictor
* mask_rcnn_box_predictor
* rfcn_box_predictor

RfcnBoxPredictor
MaskRCNNBoxPredictor
ConvolutionalBoxPredictor


### second_stage_post_processing

builders/post_processing_builder.py
proto/post_processing.proto


BatchNonMaxSuppression

ScoreConverter 支持 IDENTITY(默认),SIGMOID,SOFTMAX
