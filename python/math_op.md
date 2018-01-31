
abs(x, name=None)                       : 略
divide(x, y, name=None)                 : 略
multiply(x, y, name=None)               : 略
subtract(x, y, name=None)               : 略
negative(x, name=None)                  : 略
sign(x, name=None)                      : 略
square(x, name=None)                    : 略
sqrt(x, name=None)                      : 略
erf(x, name=None)                       : 略
scalar_mul(scalar, x)                   : 略
pow(x, y, name=None)                    : 略
complex(real, imag, name=None)          : 略
real(input, name=None)                  : 略
imag(input, name=None)                  : 略
round(x, name=None)                     : 略
cast(x, dtype, name=None)               : 将 x 转为  tensor, 之后转为 dtype 的 base_dtype
saturate_cast(value, dtype, name=None)  : 类型转换的时候，考虑的不同类型的范围
to_float(x, name="ToFloat")             :  x 转为 dtypes.float32
to_double(x, name="ToDouble")           :  x 转为 dtypes.float64
to_int32(x, name="ToInt32")             :  x 转为 dtypes.int32
to_int64(x, name="ToInt64")             :  x 转为 dtypes.int64
to_bfloat16(x, name="ToBFloat16")       :  x 转为 dtypes.bfloat16
truediv(x, y, name=None):               :  x / y (会把  x, y 先转为  float 类型)
div(x, y, name=None):                   :  x / y (遵守  python 2 语义)
floordiv(x, y, name=None)               :  x / y 向下取值 
logical_xor(x, y, name="LogicalXor")    :  x ^ y
trace(x, name=None)                     :  对角线元素之和
add_n(inputs, name=None)                :  将 inputs 中每个元素的对应元素相加
sigmoid(x, name=None):                  :  y = 1 / (1 + exp(-x))
log_sigmoid(x, name=None)               :  y = log(1 / (1 + exp(-x)))  实际为 -tf.nn.softplus(-x)
tanh(x, name=None)                      :  计算 tanh
conj(x, name=None)                      :  计算 x
reduced_shape(input_shape, axes)        :  TODO
tensordot(a, b, axes, name=None)        :  TODO


argmax(input, axis=None, name=None, dimension=None, output_type=dtypes.int64):
argmin(input, axis=None, name=None, dimension=None, output_type=dtypes.int64):
bincount(arr, weights=None, minlength=None, maxlength=None, dtype=dtypes.int32) : TODO
range(start, limit=None, delta=1, dtype=None, name="range"): 范围
accumulate_n(inputs, shape=None, tensor_dtype=None, name=None)  :  将 inputs 中每个元素的对应元素相加，与 add_n 不同的是增加了类型检查
reduce_sum(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None): 对 input_tensor 中 axis 对应的元素求和
count_nonzero(input_tensor, axis=None, keep_dims=False, dtype=dtypes.int64, name=None, reduction_indices=None): 计算非 0 的个数
reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None): 计算 input_tensor 平均值
reduce_prod(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None): 计算元素乘积
reduce_min(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None):  找到最小元素
reduce_max(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None):  找到最大元素
reduce_all(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None):  计算逻辑 and，每个元素必须为 False 或 True
reduce_any(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None):  计算逻辑 and，每个元素必须为 False 或 True
reduce_logsumexp(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None):  对元素进行 log(sum(exp(element))) 计算
matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)



cumsum(x, axis=0, exclusive=False, reverse=False, name=None)
  tf.cumsum([a, b, c], exclusive=True)  # => [0, a, a + b]
  tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
  tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
  tf.cumsum([a, b, c], exclusive=True, reverse=True)  # => [b + c, c, 0]

cumprod(x, axis=0, exclusive=False, reverse=False, name=None):

  tf.cumprod([a, b, c])  # => [a, a * b, a * b * c]
  tf.cumprod([a, b, c], exclusive=True)  # => [1, a, a * b]
  tf.cumprod([a, b, c], reverse=True)  # => [a * b * c, b * c, c]
  tf.cumprod([a, b, c], exclusive=True, reverse=True)  # => [b * c, c, 1]

* add
* subtract
* multiply
* scalar_mul
* div
* divide
* floordiv
* realdiv
* truncatediv
* floor_div
* truncatemod
* floormod
* mod
* cross
* add_n
* abs
* negative
* sign
* reciprocal
* square
* round
* sqrt
* rsqrt
* pow
* exp
* expm1
* log
* log1p
* sinh
* cosh
* asinh
* acosh
* atanh
* ceil
* floor
* maximum
* minimum
* cos
* sin
* lbeta
* tan
* acos
* asin
* atan
* atan2
* lgamma
* digamma
* erf
* erfc
* squared_difference
* igamma
* igammac
* zeta
* polygamma
* betainc
* rint
* diag
* diag_part
* trace
* transpose
* eye
* matrix_diag
* matrix_diag_part
* matrix_band_part
* matrix_set_diag
* matrix_transpose
* matmul
* norm
* matrix_determinant
* matrix_inverse
* cholesky
* cholesky_solve
* matrix_solve
* matrix_triangular_solve
* matrix_solve_ls
* qr
* self_adjoint_eig
* self_adjoint_eigvals
* svd
* tensordot
* complex
* conj
* imag
* real
* fft
* ifft
* fft2d
* ifft2d
* fft3d
* ifft3d
* reduce_sum
* reduce_prod
* reduce_min
* reduce_max
* reduce_mean
* reduce_all
* reduce_any
* reduce_logsumexp
* count_nonzero
* accumulate_n
* einsum
* bincount
* cumsum
* cumprod
* segment_sum
* segment_prod
* segment_min
* segment_max
* segment_mean
* unsorted_segment_sum
* unsorted_segment_max
* sparse_segment_sum
* sparse_segment_mean
* sparse_segment_sqrt_n
* argmin
* argmax
* setdiff1d
* where
* unique
* edit_distance
* invert_permutation
