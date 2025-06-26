
from PIL import Image
from torch import randn 
import fonctions_utiles as f
from torch.utils.data import Dataset


#____________________________________________________________________________________________________
# version où on charge seulement les images

# Créer un Dataset personnalisé pour charger les images
class CustomImageDataset (Dataset):
    """
    seulement les images en RGB
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_paths)

#____________________________________________________________________________________________________
# version où on charge seulement les images mais en nivau de gris

# Créer un Dataset personnalisé pour charger les images
class CustomImageDataset_BW (Dataset):
    """
    seulement les images en niveaux de gris
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('L')

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_paths)


#__________________________________________________________________________________________
# version où on charge les images avec leur landmarks 

# Créer un Dataset personnalisé pour charger les images et les landmarks
class CustomImageLandmarksDataset (Dataset):
    """ 
    les images RGB avec la liste des landmarks  
    """
    def __init__(self, image_paths, landmarks, transform=None):
        self.image_paths = image_paths
        self.landmarks = landmarks
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        landmarks = self.landmarks[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, landmarks

    def __len__(self):
        return len(self.image_paths)
    


#__________________________________________________________________________________________
# version où on charge les images et une version bruitée
class CustomImageNoisedDataset (Dataset):
    """  
    les images et une version bruité
    """
    def __init__(self, image_paths,  transform=None, noise = 0.1 ):
        self.image_paths = image_paths
        self.transform = transform
        self.noise_level = noise

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        
        image_noised = image + randn(image.shape)*self.noise_level 
        #torch.randn(size) donne un tenseur de taille size remplie de N(0,1)
        return image, image_noised

    def __len__(self):
        return len(self.image_paths)
    

#__________________________________________________________________________________________
# version où on charge les images avec leur landmarks : landarsk en image bin

# Créer un Dataset personnalisé pour charger les images et les landmarks
class CustomLandmarksBinaryDataset (Dataset):
    """ 
    les landmarks sous forme d'image 
    """
    def __init__(self, landmarks, transform=None):
        self.landmarks = landmarks
        self.transform = transform

    def __getitem__(self, index):
        landmarks = f.im_bin_landmarks(self.landmarks[index])

        if self.transform is not None:
            landmarks = self.transform(landmarks)
        
        return landmarks

    def __len__(self):
        return len(self.landmarks)


# Créer un Dataset personnalisé pour charger les images et les landmarks
class CustomImageLandmarksBinaryDataset (Dataset):
    """ 
    les images en niveau de gris et leslandmarks sous forme d'image
    """
    def __init__(self, image_paths, landmarks, transform=None, transform_land=None):
        self.image_paths = image_paths
        self.landmarks = landmarks
        self.transform = transform
        self.transform_land = transform_land

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('L')
        landmarks = f.im_bin_landmarks(self.landmarks[index])

        if self.transform is not None:
            image = self.transform(image)
            landmarks = self.transform_land(landmarks)
        
        return image, landmarks

    def __len__(self):
        return len(self.image_paths)
    

__all__ = [
    'CustomImageDataset',
    'CustomImageDataset_BW',
    'CustomImageLandmarksDataset',
    'CustomImageNoisedDataset',
    'CustomLandmarksBinaryDataset',
    'CustomImageLandmarksBinaryDataset',

]