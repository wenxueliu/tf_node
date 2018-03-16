



## 图片操作

from scipy.misc import imread, imsave, imresize

img = imread('assets/cat.jpg')
print(img.dtype, img.shape)

img_tinted = img * [1, 0.95, 0.9]

img_tinted = imresize(img_tinted, (300, 300))

imsave('assets/cat_tinted.jpg', img_tinted)

## MatLab 文件

scipy.io.loadmat

scipy.io.savemat

## 点之间的距离

scipy.spatial.distance.pdist

scipy.spatial.distance.cdist

from scipy.spatial.distance import pdist, squareform

x = np.array([[0, 1], [1, 0], [2, 0]])
d = squareform(pdist(x, 'euclidean'))
