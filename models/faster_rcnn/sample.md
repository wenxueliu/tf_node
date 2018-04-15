
## MinibatchSampler

MinibatchSampler
    BalancedPositiveNegativeSampler

subsample(self, indicator, batch_size, \**params)
subsample_indicator(indicator, num_samples) :  将 indicator 的顺序打乱，取前 num_samples 为 True，剩余的设置为 False

BalancedPositiveNegativeSampler

self._positive_fraction 正样本的比例

def subsample(self, indicator, batch_size, labels)

输入
* indicator : [N] 样本索引
* labels : [][][N] bool 标记 indicator 哪些是正样本哪些是负样本
* batch_size :  正负样本的总数量

输出
正负样本索引

1. 根据 labels 从 indicator 中取到正样本索引 positive_idx 和负样本的索引 negative_idx
2. 从 positive_idx 打乱顺序，取前 batch_size * self._positive_fraction 个元素作为正样本
3. 从 negative_idx 打乱顺序，取前 batch_size * (1 - self._positive_fraction) 个元素作为负样本
4. 将正样本或负样本设置为 True, 其余设置为 False
