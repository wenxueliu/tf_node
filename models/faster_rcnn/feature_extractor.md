

## FeatureExtractor

FeatureExtractor
    FasterRCNNFeatureExtractor
        FasterRCNNResnetV1FeatureExtractor
            FasterRCNNResnet50FeatureExtractor
            FasterRCNNResnet101FeatureExtractor
            FasterRCNNResnet152FeatureExtractor
        FasterRCNNInceptionV2FeatureExtractor
        FasterRCNNInceptionResnetV2FeatureExtractor

* FasterRCNNNASFeatureExtractor

#### FasterRCNNFeatureExtractor

* self._is_training
* self._first_stage_features_stride
* self._train_batch_norm
* self._reuse_weights
* self._weight_decay

#### FasterRCNNInceptionV2FeatureExtractor

self._depth_multiplier = depth_multiplier
self._min_depth = min_depth

first_stage_features_stride 只支持 8 或 16

预处理

(2.0 / 255.0) * resized_inputs - 1.0

特征提取

extract_proposal_features : inception_v2.inception_v2_base


#### FasterRCNNResnetV1FeatureExtractor

self._architecture
self._resnet_model

预处理

resized_inputs - [[123.68, 116.779, 103.939]]

特征提取

* FasterRCNNResnet50FeatureExtractor : resnet_v1.resnet_v1_50
* FasterRCNNResnet101FeatureExtractor : resnet_v1.resnet_v1_101
* FasterRCNNResnet152FeatureExtractor : resnet_v1.resnet_v1_152


FasterRCNNInceptionResnetV2FeatureExtractor

预处理

(2.0 / 255.0) * resized_inputs - 1.0

特征提取

inception_resnet_v2.inception_resnet_v2_base
