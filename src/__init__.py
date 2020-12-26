import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import math
import colorsys

img_path = '../images/rivendell.jpg'
clusters = 10
debug = False

# img = cv2.imread('../images/valley.jpg')
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
# plt.show()


#####
# Considerando cada cor 

if (debug):
    array = np.zeros([256,256,256])

    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            pixel = img[row][column]
            array[pixel[0]][pixel[1]][pixel[2]] = array[pixel[0]][pixel[1]][pixel[2]] + 1

    z,x,y = array.nonzero()
    colors = list(zip(z/255, x/255, y/255))
    weight = [ 10*math.log(array[i[0]][i[1]][i[2]]) for i in list(zip(z, x, y))]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z, x, y, zdir='z', c=colors, s=weight)
    plt.show()

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



# Considerando cada pixel 

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

if (debug):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r, g, b, zdir='r', c=colors)
    plt.show()

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

if (debug):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r, g, b, zdir='r', c=clustered_colors,)
    plt.show()

plt.imshow(img)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x=list(range(len(centroids))), height=[1] * len(centroids),  color=[np.array(centroid, int)/255 for centroid in centroids])
plt.show()

## Ordenando cores

ordered_centroids = list(centroids)
ordered_centroids.sort(key=lambda rgb: colorsys.rgb_to_hls(*rgb))
centroids = ordered_centroids

##

## OUTPUT

result = ''

with open("output_template.html", "r") as template:
    result = template.read()

result = result.replace('@image', img_path)
result = result.replace('@width_plus_padding', str(img.shape[1] + 10))
content = [f'<td style="height: {int(0.40 * img.shape[0])};  background-color: rgb{int(c[0]), int(c[1]), int(c[2])};"></td>' for c in centroids]
result = result.replace('@content', ''.join(content))

with open("../output/output.html", "w") as output:
    output.truncate()
    output.write(result)

##

