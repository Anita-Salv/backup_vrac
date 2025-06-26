import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from scipy.spatial.distance import pdist, squareform


#____________________ Calcul du niveau de gris d'une image dans son coin supérieur gauche
#__________________________________________________________________________________________
def niveau_gris(img, taille = 10):
    niv = 0
    for i in range(taille):
        for j in range(taille):
            niv += (img[0,i,j] + img[1,i,j] + img[2,i,j])/3
    return (niv/(taille**2)).item()

def niveau_gris_moyen(img):
    return niveau_gris(img, taille = len(img[0][0]))

def niveau_gris_BW(img, taille = 10):
    niv = 0
    for i in range(taille):
        for j in range(taille):
            niv += img[0][i][j]/(taille**2)
    return (niv).item()

def niveau_gris_moyen_BW(img):
    return niveau_gris_BW(img, taille = len(img[0]))


#____________________ transforme la liste des landmarks en image avec des points à l'emplacemnt des landmark
#__________________________________________________________________________________________
def im_bin_landmarks(landmarks, size =(1, 218, 178) ):
    img = torch.zeros(size=size)
    decal = [-2, -1, 0, 1, 2]
    for j in range(0,10,2):
        x = int(landmarks[j])
        y = int(landmarks[j+1])
        for k in decal:
            for l in decal:
                img[0][y+k][x+l] = 1

    return img


#____________________ pour afficher simplement des images

def plot_img(point, axis = 'off'):
     plt.imshow(np.transpose(point, (1,2,0)))
     plt.axis(axis)

def plot_img_bw(point):
    plt.imshow(point.squeeze(), cmap = 'gray')
    plt.axis('off')

def plot_img_with_landmarks(img, landmarks):
    plot_img(img)
    for j in range(0,10,2):
        plt.plot(landmarks[j],landmarks[j+1], '*', color='r')
    plt.axis('equal')
    plt.axis('off')
    

#____________________ calculs desindices à partir des landmarks
#__________________________________________________________________________________________
def orientation(pt):
    #position droite/gauche du nez normalisé : 
    #  O au centre, -1 au niveau de l'oeil gauche, 1 au niveau de l'oeil droit
    nose_x = pt[1][4] #entre 0 et 178
    left_eye_x = pt[1][0]
    right_eye_x = pt[1][2]
    eccart = right_eye_x - left_eye_x
    return (2/eccart)*(nose_x - right_eye_x) + 1

def sourire(pt):
    #distance en pixel entre les des deux coins de la bouche : 
    left_mouth = pt[1][6:8]
    right_mouth = pt[1][8:10]
    return np.linalg.norm(left_mouth - right_mouth)


# ____________________récupérer les données
#__________________________________________________________________________________________
def get_data(subset, shuffle=True, batch_size=32):
    """ Extracts data from a Subset torch dataset in the form of a tensor"""
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    return next(iter(loader))


#________________________print pairplot 
#__________________________________________________________________________________________
def print_pairplot_mcvae(array_z, 
                   LIST_DIM_PAIR_PLOT=[0,1,2], 
                   indice_color=None,
                   marker = ',',
                   s = 1,
                   cmap = 'jet', 
                   best_points = None, 
                   col_best = 'forest_green',
                   worst_points = None,
                   col_worst = 'crimson',
                   alpha = None,
                   colbar=True,
                   indice=None
                   ):
    n_chan = len(array_z[0])
    nb_dim = len(LIST_DIM_PAIR_PLOT)
    _, axes = plt.subplots(nb_dim, nb_dim, 
                             figsize=(5*nb_dim, 5*nb_dim))
    ax_list = axes.ravel()

    

    if best_points is None :
        for i, a in enumerate(LIST_DIM_PAIR_PLOT):
            for j in range(i+1):
                o = LIST_DIM_PAIR_PLOT[j]
                plt.subplot(nb_dim, nb_dim, nb_dim*i+j+1) 
                
                if indice_color is None:
                    list_col=['r','g','b']
                    for k in range(n_chan):
                        plt.scatter(array_z[:,k,a], array_z[:,k,o], s=1,  
                                    c=list_col[k] )#indice_color, cmap = cmap)
                else : 
                    for k in range(n_chan):
                        plt.scatter(array_z[:,k,a], array_z[:,k,o], s=1,  
                                    c=indice_color, cmap = cmap)
                plt.xlabel(f"dim {a}")
                plt.ylabel(f"dim {o}")
        
        if colbar :
            plt.colorbar(ax=axes,  shrink = 0.6)
            name_indice = indice.__name__
            plt.suptitle(f"mc_vae coloration selon {name_indice}", 
                    verticalalignment = 'bottom', 
                    size = 'xx-large')

    if best_points is not None : 
        for i, a in enumerate(LIST_DIM_PAIR_PLOT):
            for j in range(i+1):
                o = LIST_DIM_PAIR_PLOT[j]
                plt.subplot(nb_dim, nb_dim, nb_dim*i+j+1) 
                for k in range(n_chan):
                    plt.scatter(array_z[:,k,a], array_z[:,k,o], s=s, marker = marker, c= 'grey', alpha = alpha)
                    plt.scatter(best_points[:,k,a], best_points[:,k,o], s=20, marker = '*',  color = col_best, label = 'best points from test')
                    plt.scatter(worst_points[:,k,a], worst_points[:,k,o], s=20, marker = '*',  color = col_worst, label = 'worst points from test')
                plt.xlabel(f"dim {a}")
                plt.ylabel(f"dim {o}")

    for i in range(nb_dim):
        for j in range(i+1,nb_dim):
            plt.subplot(nb_dim, nb_dim, nb_dim*i+j+1)
            plt.axis('off')


#__________________________________________________________________________________________
# on affiche toutes les points selon toutes les paires de dimension possible
def print_pairplot(array, 
                   LIST_DIM_PAIR_PLOT=[0,1,2], 
                   indice_color=None,
                   marker = ',',
                   s = 1,
                   cmap = 'jet', 
                   best_points = None, 
                   col_best = 'forest_green',
                   worst_points = None,
                   col_worst = 'crimson',
                   alpha = None,
                   indice = None
                   ):
    nb_dim = len(LIST_DIM_PAIR_PLOT)
    fig, axes = plt.subplots(nb_dim, nb_dim, 
                            figsize=(5*nb_dim, 5*nb_dim))
    ax_list = axes.ravel()

    if indice_color is None:
        indice_color = np.zeros(len(array))

    for i, a in enumerate(LIST_DIM_PAIR_PLOT):
        for j in range(i+1):
            o = LIST_DIM_PAIR_PLOT[j]
            plt.subplot(nb_dim, nb_dim, nb_dim*i+j+1)
            if best_points is None : 
                plt.scatter(array[:,a], array[:,o], s=s, marker = marker, c= indice_color, cmap = cmap)
            if best_points is not None : 
                plt.scatter(array[:,a], array[:,o], s=s, marker = marker, c= 'grey', alpha = alpha)
                plt.scatter(best_points[:,a], best_points[:,o], s=20, marker = '*',  color = col_best, label = 'best points from test')
                plt.scatter(worst_points[:,a], worst_points[:,o], s=20, marker = '*',  color = col_worst, label = 'worst points from test')
            plt.xlabel(f"dim {a}")
            plt.ylabel(f"dim {o}")

    if indice is not None : 
        plt.colorbar(ax=axes,  shrink = 0.6)
        name_indice = indice.__name__
        plt.suptitle(f"color according to {name_indice}", 
                    verticalalignment = 'bottom', 
                    size = 'xx-large')
        

    for i in range(nb_dim):
        for j in range(i+1,nb_dim):
            plt.subplot(nb_dim, nb_dim, nb_dim*i+j+1)
            plt.axis('off')


# ____________________calcul matrice de ditance / affinité 
#__________________________________________________________________________________________

def dist_matrix (array, norm = np.linalg.norm):
    n = len(array)
    matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            d = norm(array[i]- array[j])
            matrix[i,j] = d
            matrix[j,i] = d
    return matrix 

def gauss_norm(matrix, sig=None):
    if sig is None : 
        sig = matrix.std()
    return np.exp( -(matrix )**2 / (2*sig**2) )

def dist_moy_k_voisin(list_dist, k=5):
    sorted = np.sort(list_dist)
    i=0
    while sorted[i]==0:
        i+=1
    return np.mean(sorted[i:i+k])

def moy_dist_k_voisin(mat_dist, k=5):
    N = len(mat_dist)
    dists = np.zeros(N)
    for i in range(N):
        dists[i] = dist_moy_k_voisin(mat_dist[i], k=k)
    return np.mean(dists)

def mat_aff_trie(x, kNN = 5, sigma = None):
    N = len(x)
    # Calculer la matrice des distances (équivalent de pdist et squareform)
    tmpK = squareform(pdist(x))


    if sigma is None : 
        # Supprimer les diagonales (self-distances) en les remplaçant par l'infini
        tmpD = tmpK + np.diag(np.full(N, np.inf))

        # Trier les distances et prendre les k plus proches voisins (kNN)
        tmpD = np.sort(tmpD, axis=0)
        tmpB = tmpD[:kNN, :]

        # Calcul de sigma comme la moyenne des distances des k plus proches voisins
        sigma = np.mean(tmpB)

    # Trier les points sur la première dimension
    xS = np.sort(x, axis=0)

    # Calculer la matrice des distances pour les points triés
    tmpS = squareform(pdist(xS))

    # Calculer la matrice d'affinité pour les points triés
    KS = np.exp(-tmpS**2 / (2 * sigma**2))

    return KS

def mat_dist(x):
    return squareform(pdist(x))

def mat_aff(x, kNN=5, sigma=None):
    N = len(x)
    # Calculer la matrice des distances (équivalent de pdist et squareform)
    tmpK = squareform(pdist(x))

    if sigma is None : 
        # Supprimer les diagonales (self-distances) en les remplaçant par l'infini
        tmpD = tmpK + np.diag(np.full(N, np.inf))

        # Trier les distances et prendre les k plus proches voisins (kNN)
        tmpD = np.sort(tmpD, axis=0)
        tmpB = tmpD[:kNN, :]

        # Calcul de sigma comme la moyenne des distances des k plus proches voisins
        sigma = np.mean(tmpB)

    # Calculer la matrice d'affinité avec un noyau gaussien
    K = np.exp(-tmpK**2 / (2 * sigma**2))

    return K


#________________________ plot d'une mtrice contre une autre (nuage de points)
#__________________________________________________________________________________________

def plot_matx_maty (matx, maty, 
                    alpha = .3, 
                    title=None, 
                    xlab = ' ', 
                    ylab = ' ', 
                    color_pt = 'b', 
                    size_pt=.5, 
                    m=1, 
                    axis=None,
                    color_diag='r',
                    diag = True,):
    nb_pt = len(matx)
    for i in range(nb_pt):
        plt.scatter(matx[i], maty[i], color=color_pt, 
                    marker = ',', edgecolors='none', s=size_pt)
    if diag : 
        plt.plot(np.linspace(0,m, 100),np.linspace(0,m, 100), color=color_diag)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if title is None :
        title = ylab +' contre '+xlab
    plt.title(title)
    plt.axis(axis)


#________________________ plot de deux nuages de points et les segments reliants les deux nuages
#__________________________________________________________________________________________

def plot_pts_reliés(x, y, 
                    dim_abs = 0,
                    dim_ord = 1,
                    colx = 'b', 
                    coly = 'g', 
                    col_lien = 'k', 
                    alpha_lien = .3, 
                    alphax= 1, 
                    alphay = 1, 
                    style_lien = '--',
                    title=None,
                    grid=True,
                    axis = 'equal' #None to do it automaticaly
                    ):
    # Affichage des points
    N = len(x)
    plt.plot(x[:, dim_abs], x[:, dim_ord], '.', alpha = alphax, color = colx)
    plt.plot(y[:, dim_abs], y[:, dim_ord], '.', alpha = alphay, color = coly)
    for i in range(N):
        plt.plot([x[i, dim_abs], y[i, dim_abs]], [x[i, dim_ord], y[i, dim_ord]], style_lien, alpha = alpha_lien, color=col_lien)
    plt.xlabel(f'dim {dim_abs}')
    plt.ylabel(f'dim {dim_ord}')
    plt.axis(axis)
    plt.grid(grid)
    plt.title(title)



def print_pairplot_align_relie(x, y, alpha_pt = .4, LIST_DIM_PAIR_PLOT = [0,1,2], alpha_lien=.5,
                         colx = 'b', 
                    coly = 'g', 
                    col_lien = 'k',
                    style_lien = '--', ):
    nb_dim = len(LIST_DIM_PAIR_PLOT)

    plt.figure(figsize=(5*nb_dim, 5*nb_dim))

    for i, a in enumerate(LIST_DIM_PAIR_PLOT):
                for j in range(i):
                    o = LIST_DIM_PAIR_PLOT[j]
                    plt.subplot(nb_dim, nb_dim, nb_dim*i+j+1) 
                    plot_pts_reliés(x, y, 
                                    dim_abs=a, dim_ord=o, 
                                    colx = colx, 
                                    coly = coly, 
                                    col_lien = col_lien, 
                                    style_lien = style_lien,
                                    alphax=alpha_pt, 
                                    alpha_lien = alpha_lien, 
                                    alphay=alpha_pt,
                                    axis=None)

    for i in range(nb_dim):
            for j in range(i,nb_dim):
                plt.subplot(nb_dim, nb_dim, nb_dim*i+j+1)
                plt.axis('off')