import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

imagefile = r'C:\Users\salvador\Downloads\train-images-idx3-ubyte\train-images.idx3-ubyte'
labelfile = r'C:\Users\salvador\Downloads\train-labels-idx1-ubyte\train-labels.idx1-ubyte'

imagearray = idx2numpy.convert_from_file(imagefile)
labelarray = idx2numpy.convert_from_file(labelfile)

def affiche (im):
    plt.imshow(im, cmap=plt.cm.binary)

def affiche_im_moy(list_im):
    affiche(np.mean(list_im, axis=0))

ind_start = 518
nb_im = 8
nb_col = 4
nb_lig = 4
# On veut afficher toutes les images plus leur moyenne
# on doit donc avoir nb_im +1 < nb_col*nb_lig

plt.figure()
for i in range(nb_im):
    plt.subplot(nb_lig, nb_col, i+1)
    affiche(imagearray[ind_start+i])
    plt.title(labelarray[ind_start+i])
plt.subplot(nb_col, nb_lig, nb_im+1)
affiche_im_moy(imagearray[ind_start:ind_start+nb_im])
plt.show()

def add(x,y):
    return x+y