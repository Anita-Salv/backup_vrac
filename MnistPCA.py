import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

#On importe sous forme d'un nparray de taille 6000 x 28 x 28 6000 images de Mnist
imagefile = r'C:\Users\salvador\Downloads\train-images-idx3-ubyte\train-images.idx3-ubyte'
labelfile = r'C:\Users\salvador\Downloads\train-labels-idx1-ubyte\train-labels.idx1-ubyte'

image_array = idx2numpy.convert_from_file(imagefile)
label_array = idx2numpy.convert_from_file(labelfile)


def affiche (im):
    plt.imshow(im, cmap=plt.cm.binary)

from sklearn.decomposition import PCA

# paramètre : nombre d'images et lesquelle on choisit
ind_depart = 2000
nb_points = 2000
taille_im = 28*28 #(=784)
nb_dim_pca = 100

#On construit un nparray contenant les images mais flatten 
# pour avoir des vecteur de taille 28*28 et pas des matrices 2D
X = np.zeros((nb_points,taille_im))

for (i,im) in enumerate(image_array[ind_depart:ind_depart+nb_points]):
    X[i]=im.flatten()
A =X[0]
pca = PCA(n_components=nb_dim_pca)
pca.fit(X)
Y = pca.transform(X)

var_expl = pca.explained_variance_ratio_
sing_vals = pca.singular_values_


list_colors=['red', 'blue', 'green', 'pink', 'purple', 'grey', 'teal', 'black', 'orange', 'deepskyblue']

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(sing_vals)
plt.title('singuliars values')

plt.subplot(2,2,2)
plt.plot(var_expl)
plt.title('variance explained')

plt.subplot(2,2,3)
#on trace les axes
plt.plot(np.linspace(np.min(Y[:,0]), np.max(Y[:,0]), 100), np.zeros(100), color = 'k')
plt.plot( np.zeros(100), np.linspace(np.min(Y[:,1]), np.max(Y[:,1]), 100), color = 'k')
plt.plot(Y[:,0], Y[:,1], '*')
plt.title('nuage point entre dim0 et dim1')

plt.subplot(2,2,4)
list_lab=[]
 #on trace les axes
plt.plot(np.linspace(np.min(Y[:,0]), np.max(Y[:,0]), 100), np.zeros(100), color = 'k')
plt.plot( np.zeros(100), np.linspace(np.min(Y[:,1]), np.max(Y[:,1]), 100), color = 'k')
for (i,el) in enumerate(Y):
    lab = label_array[ind_depart+i]
    if lab in list_lab : 
        plt.plot(el[0], el[1], '.', color = list_colors[lab])
    else : 
        plt.plot(el[0], el[1], '.', color = list_colors[lab], label=str(lab))
        list_lab.append(lab)
plt.title('nuage points entre dim0 et dim1 color by label')
plt.legend()

plt.show()