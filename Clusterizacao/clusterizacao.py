#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

# Cria um array de tres camadas, cada camada é pra um valor do RGB R G B
img_arr = img.imread('paisagem.bmp')

plt.imshow(img_arr)

(heigth,width,qtd_colors) = img_arr.shape

img2D = img_arr.reshape(heigth*width, qtd_colors)

# Kmeans randomico

from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=4, init="random")

clusters_labels = kmeans_model.fit_predict(img2D)

centroid = kmeans_model.cluster_centers_

rgb_colors = centroid.round(0).astype(int)

labels = (kmeans_model.labels_) # Para qual grupo o pixel pertence

img_quant = np.reshape(rgb_colors[clusters_labels], (heigth, width, qtd_colors))

plt.imshow(img_quant)

#############

labels=list(kmeans_model.labels_)
percent=[]

for i in range(len(centroid)):
    j=labels.count(i)
    j=j/(len(labels))
    percent.append(j)
    
print(percent)

fig, ax = plt.subplots(1,3, figsize=(20,12))

ax[0].imshow(img_arr)
ax[0].set_title('Imagem Original')
ax[1].imshow(img_quant)
ax[1].set_title('Imagem Quantizada')
ax[2].pie(percent,colors=np.array(centroid/255),labels=np.arange(len(centroid)))
ax[2].set_title('Paleta de cores')


# Kmeans++

from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=4, init="k-means++")

clusters_labels = kmeans_model.fit_predict(img2D)

centroid = kmeans_model.cluster_centers_

rgb_colors = centroid.round(0).astype(int)

labels = (kmeans_model.labels_) # Para qual grupo o pixel pertence

img_quant = np.reshape(rgb_colors[clusters_labels], (heigth, width, qtd_colors))

plt.imshow(img_quant)

#############

labels=list(kmeans_model.labels_)
percent=[]

for i in range(len(centroid)):
    j=labels.count(i)
    j=j/(len(labels))
    percent.append(j)
    
print(percent)

fig, ax = plt.subplots(1,3, figsize=(20,12))

ax[0].imshow(img_arr)
ax[0].set_title('Imagem Original')
ax[1].imshow(img_quant)
ax[1].set_title('Imagem Quantizada')
ax[2].pie(percent,colors=np.array(centroid/255),labels=np.arange(len(centroid)))
ax[2].set_title('Paleta de cores')


# Kmeans++ - Com inicializacao nos centroids

centroids = []
centroids.append([52, 177, 190]) # Agua
centroids.append([251, 231, 200]) # Areia
centroids.append([18, 10, 35]) # Montanha
centroids.append([255, 255, 255]) # Agua

centroids=np.array(centroids)

from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=4, init=centroids)

clusters_labels = kmeans_model.fit_predict(img2D)

centroid = kmeans_model.cluster_centers_

rgb_colors = centroid.round(0).astype(int)

labels = (kmeans_model.labels_) # Para qual grupo o pixel pertence

img_quant = np.reshape(rgb_colors[clusters_labels], (heigth, width, qtd_colors))

plt.imshow(img_quant)

#############

labels=list(kmeans_model.labels_)
percent=[]

for i in range(len(centroid)):
    j=labels.count(i)
    j=j/(len(labels))
    percent.append(j)
    
print(percent)

fig, ax = plt.subplots(1,3, figsize=(20,12))

ax[0].imshow(img_arr)
ax[0].set_title('Imagem Original')
ax[1].imshow(img_quant)
ax[1].set_title('Imagem Quantizada')
ax[2].pie(percent,colors=np.array(centroid/255),labels=np.arange(len(centroid)))
ax[2].set_title('Paleta de cores')