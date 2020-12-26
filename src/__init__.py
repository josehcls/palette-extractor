import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import math

img = cv2.imread('../images/kagemusha.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
# plt.show()

clusters = 10

#####

# array = np.zeros([256,256,256])

# for row in range(img.shape[0]):
#     for column in range(img.shape[1]):
#         pixel = img[row][column]
#         array[pixel[0]][pixel[1]][pixel[2]] = array[pixel[0]][pixel[1]][pixel[2]] + 1

# z,x,y = array.nonzero()
# colors = list(zip(z/255, x/255, y/255))
# weight = [ 10*math.log(array[i[0]][i[1]][i[2]]) for i in list(zip(z, x, y))]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(z, x, y, zdir='z', c=colors, s=weight)
# plt.show()

# kmeans = KMeans(
#     n_clusters=clusters,
#     init='random',
#     n_init=10,
#     max_iter=300,
#     tol=1e-04,
#     random_state=0
# )

# array_ = [ i for i in list(zip(z, x, y))]
# labels = kmeans.fit_predict(array_)
# centroids  = kmeans.cluster_centers_ 

# available_colors = {}
# for i in range(len(centroids)):
#     available_colors[i] = (centroids[i][0]/255, centroids[i][1]/255, centroids[i][2]/255)

# colors_ = [available_colors[i] for i in labels]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(z, x, y, zdir='z', c=colors_,)
# plt.show()

# Considerar peso criando multiplos pontos ??


width = img.shape[0]
heigth = img.shape[1]
r = []
g = []
b = []

for i in range(width):
    for j in range(heigth):
        r.append(img[i][j][0])
        g.append(img[i][j][1])
        b.append(img[i][j][2])

colors = list(zip(np.array(r)/255, np.array(g)/255, np.array(b)/255))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(r, g, b, zdir='r', c=colors)
# plt.show()

kmeans = KMeans(
    n_clusters=clusters,
    init='random',
    n_init=10,
    max_iter=300,
    tol=1e-04,
    random_state=0
)
print('Fitting...', end='')

array_ = [ i for i in list(zip(r, g, b))]
labels = kmeans.fit_predict(array_)
centroids  = kmeans.cluster_centers_ 
print('Done!')

available_colors = {}
for i in range(len(centroids)):
    available_colors[i] = (centroids[i][0]/255, centroids[i][1]/255, centroids[i][2]/255)

clustered_colors = [available_colors[i] for i in labels]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(r, g, b, zdir='r', c=clustered_colors,)
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x=list(range(len(centroids))), height=[1] * len(centroids),  color=[np.array(centroid, int)/255 for centroid in centroids])
plt.show()
