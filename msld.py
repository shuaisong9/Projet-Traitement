import os
from typing import List, Tuple
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import convolve
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import math as math
from copy import deepcopy
import skimage.morphology as skmorph

class MSLD:
    """
    Classe implémentant l'algorithme de MSLD, ainsi que différents outils de
    mesure de performances.

    Les attributs de cette classe sont:
        W: Taille de la fenêtre.
        L: Vecteur contenant les longueurs des lignes à détecter.
        n_orientation: Nombre d'orientation des lignes à détecter.
        threshold: Seuil de segmentation (à apprendre).
        line_detectors_masks: Masques pour la détection des lignes pour chaque valeur de L et chaque valeur de
            n_orientation.
        avg_mask: Masque moyenneur de taille W x W.
    """

    def __init__(self, W: int, L: List[int], n_orientation: int) -> None:
        """Constructeur qui initialise un objet de type MSLD. Cette méthode est appelée par
        >>> msld = MSLD(W=..., L=..., n_orientation=...)

        Args:
            W (int): Taille de la fenêtre (telle que définie dans l'article).
            L (List[int]): Une liste contenant les valeurs des longueurs des lignes qui seront détectées par la MSLD.
            n_orientation (int): Nombre d'orientations des lignes à détecter.
        """
        self.W = W
        self.L = L
        self.n_orientation = n_orientation

        self.threshold = 0.5

        # TODO: I.Q1
        self.avg_mask = np.ones((W, W))/(W*W)
        
        # TODO: I.Q2
        # line_detectors_masks est un dictionnaire contenant les masques
        # de détection de ligne pour toutes les échelles contenues
        # dans la liste L et pour un nombre d'orientation égal à
        # n_orientation. Ainsi pour toutes les valeurs de L:
        # self.line_detectors_masks[l] est une matrice de la forme [l,l,n_orientation]

        self.line_detectors_masks = {}
        for l in L:
            # On calcule le détecteur de ligne initial de taille l (les dimensions du masque sont lxl).
            
            line_detector = np.zeros((l, l))
            l_middle = math.floor(l/2)
            line_detector[l_middle, :] = 1/l

            # On initialise la liste des n_orientation masques de taille lxl.
            line_detectors_masks = [line_detector]
            # On effectue n_orientation-1 rotations du masque line_detector.

            for n in range (n_orientation-1):
                # n part de 0, mais on veut un angle non nul dès la première itération : donc n+1
                angle = (n+1)*(180 / n_orientation)

                r = cv2.getRotationMatrix2D((l//2, l//2), angle, 1)
                rotated_mask = cv2.warpAffine(line_detector, r, (l, l))
                line_detectors_masks.append(rotated_mask/rotated_mask.sum())

            # On assemble les n_orientation masques ensemble:
            self.line_detectors_masks[l] = np.stack(line_detectors_masks, axis=2)


    ############################################################################
    #                          MSLD IMPLEMENTATION                             #
    ############################################################################
    def basic_line_detector(self, grey_lvl: np.ndarray, L: int) -> np.ndarray:
        """Applique l'algorithme Basic Line Detector sur la carte d'intensité grey_lvl avec des lignes de longueurs L.

        Args:
            grey_lvl (np.ndarray): Carte d'intensité 2D avec dtype float sur laquelle est appliqué le BLD.
            L (int): Longueur des lignes (on supposera que L est présent dans self.L et donc que
                self.line_detectors_masks[L] existe).
        
        Returns:
            R (np.ndarray): Carte de réponse 2D en float du Basic Line Detector.
        """

        # TODO: I.Q4
        # Les masques de détections de lignes de longueur L initialisés dans le constructeur sont accessibles par:
        # self.line_detectors_masks[L]
        line_detector = self.line_detectors_masks[L]
        conv_img_line = np.ones((grey_lvl.shape[0], grey_lvl.shape[1], self.n_orientation))

        for n in range(self.n_orientation):
            conv_img_line[:, :, n] = convolve(grey_lvl, line_detector[:, :, n])

        # Trouver le maximum selon la 3e dimension du array 3D (axis =2)
        l_I_max = np.max(conv_img_line, 2)

        w_I_avg = convolve(grey_lvl, self.avg_mask)

        R_w = l_I_max - w_I_avg
        R_w_norm = (R_w - np.mean(R_w)) / np.std(R_w)

        R = R_w_norm

        return R

    def multi_scale_line_detector(self, image: np.ndarray) -> np.ndarray:
        """Applique l'algorithme de Multi-Scale Line Detector et combine les réponses des BLD pour obtenir la carte 
        d'intensité de l'équation 4 de la section 3.3 Combination Method.

        Args:
            image (np.ndarray): Image RGB aux intensitées en float comprises entre 0 et 1 et de dimensions
                (hauteur, largeur, canal) (canal: R=1 G=2 B=3)
        
        Returns:
            Rcombined (np.ndarray): Carte d'intensité combinée.
        """

        # TODO: I.Q6
        # Pour les hyperparamètres L et W utilisez les valeurs de self.L et self.W.

        I_igc = 1 - image[:, :, 1] # Canal vert inversé
        n_L = len(self.L)
        R_l_w = np.zeros_like(I_igc)

        for l in self.L:
            R_l_w = R_l_w + self.basic_line_detector(image[:, :, 1], l)

        Rcombined = (1/(n_L+1))*(R_l_w + I_igc)

        return Rcombined

    def learn_threshold(self, dataset: List[dict]) -> Tuple[float, float]:
        """
        Apprend le seuil optimal pour obtenir la précision la plus élevée
        sur le dataset donné.
        Cette méthode modifie la valeur de self.threshold par le seuil
        optimal puis renvoie ce seuil et la précision obtenue.
        
        Args:
            dataset (List[dict]): Liste de dictionnaires contenant les champs ["image", "label", "mask"].
            
        Returns:
            threshold (float): Seuil proposant la meilleure précision
            accuracy (float): Valeur de la meilleure précision
         """
        # fpr : false positive (vecteur float)
        # tpr : true positive (vecteur float)
        # thresholds : vecteur float des seuils associés à ces taux

        fpr, tpr, thresholds = self.roc(dataset)

        # TODO: I.Q10
        # Utilisez np.argmax pour trouver l'indice du maximum d'un vecteur.

        # calculer P, N, et S
        # utiliser mask pour prendre en compte l'oeil seulement
        # label = bool déjà
        # en argument, on met train ou test de la fonction load_dataset

        P_somme = 0
        S_somme = 0

        for d in dataset:
            # Pour chaque élément de dataset
            label = d["label"]
            mask = d["mask"]

            retine = label[mask]
            P = np.sum(retine)
            S = np.sum(mask)

            P_somme += P
            S_somme += S

        N_somme = S_somme - P_somme

        accuracies = (tpr*P_somme + N_somme*(1-fpr))/(S_somme)
        accuracy_max_ind = np.argmax(accuracies)
        accuracy = np.amax(accuracies)
        threshold = thresholds[accuracy_max_ind]

        self.threshold = threshold

        return threshold, accuracy

    def segment_vessels(self, image: np.ndarray) -> np.ndarray:
        """
        Segmente les vaisseaux sur une image en utilisant la MSLD.
        
        Args:
            image (np.ndarray): Image RGB sur laquelle appliquer l'algorithme MSLD.

        Returns:
            vessels (np.ndarray): Carte binaire 2D de la segmentation des vaisseaux.
        """

        # TODO: I.Q13
        # Utilisez self.multi_scale_line_detector(image) et self.threshold.
        img = self.multi_scale_line_detector(image)
        vessels = np.zeros_like(img)
        vessels = img > self.threshold

        return vessels

    ############################################################################
    #                           Visualisation                                  #
    ############################################################################
    def show_diff(self, sample: dict) -> None:
        """Affiche la comparaison entre la prédiction de l'algorithme et les valeurs attendues (labels) selon le code
        couleur suivant:
           - Noir: le pixel est absent de la prédiction et du label
           - Rouge: le pixel n'est présent que dans la prédiction
           - Bleu: le pixel n'est présent que dans le label
           - Blanc: le pixel est présent dans la prédiction ET le label

        Args:
            sample (dict): Un échantillon provenant d'un dataset contenant les champs ["data", "label", "mask"].
        """

        # Calcule la segmentation des vaisseaux
        pred = self.segment_vessels(sample["image"])

        # Applique le masque à la prédiction et au label
        pred = pred & sample["mask"]
        label = sample["label"] & sample["mask"]

        # Calcule chaque canal de l'image:
        # rouge: 1 seulement pred est vrai, 0 sinon
        # bleu: 1 seulement si label est vrai, 0 sinon
        # vert: 1 seulement si label et pred sont vrais (de sorte que la couleur globale soit blanche), 0 sinon
        red = pred * 1.0
        blue = label * 1.0
        green = (pred & label) * 1.0

        rgb = np.stack([red, green, blue], axis=2)
        plt.imshow(rgb)
        plt.axis("off")
        plt.title("Différences entre la segmentation prédite et attendue")

    ############################################################################
    #                         Segmentation Metrics                             #
    ############################################################################
    def roc(self, dataset: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule la courbe ROC de l'algorithme MSLD sur un dataset donné et sur la région d'intérêt indiquée par le 
        champ "mask".

        Parameters:
            dataset (List[dict]): Base de données sur laquelle calculer la courbe ROC.

        Returns:
            fpr (np.ndarray): Vecteur float des taux de faux positifs.
            tpr (np.ndarray): Vecteur float des taux de vrais positifs.
            thresholds (np.ndarray): Vecteur float des seuils associés à ces taux.
        """

        y_true = []
        y_pred = []

        for d in dataset:
            # Pour chaque élément de dataset
            label = d["label"]  # On lit le label
            mask = d["mask"]  # le masque
            image = d["image"]  # et l'image de l'élément.

            # On calcule la prédiction du msld sur cette image.
            prediction = self.multi_scale_line_detector(image)

            # On applique les masques à label et prediction pour qu'ils contiennent uniquement
            # la liste des pixels qui appartiennent au masque.
            label = label[mask]
            prediction = prediction[mask]

            # On ajoute les vecteurs label et prediction aux listes y_true et y_pred
            y_true.append(label)
            y_pred.append(prediction)

        # On concatène les vecteurs de la listes y_true pour obtenir un unique vecteur contenant
        # les labels associés à tous les pixels qui appartiennent au masque du dataset.
        y_true = np.concatenate(y_true)
        # Même chose pour y_pred.
        y_pred = np.concatenate(y_pred)

        # On calcule le taux de vrai positif et de faux positif du dataset pour chaque seuil possible.
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        return fpr, tpr, thresholds

    def naive_metrics(self, dataset: List[dict]) -> Tuple[float, np.ndarray]:
        """
        Évalue la précision et la matrice de confusion de l'algorithme sur
        un dataset donné et sur la région d'intérêt indiquée par le
        champs mask.

        Args:
            dataset (List[dict]): Base de données sur laquelle calculer les métriques.
            aka train_erroded

        Returns:
            accuracy (float): Précision.
            confusion_matrix (np.ndarray): Matrice de confusion 2 x 2 normalisée par le nombre de labels positifs et
                négatifs.
        """

        # TODO: II.Q1

        tp = 0 # Nombre total de true positive
        fp = 0 # Nombre total de false positive
        fn = 0 # Nombre total de false negative
        tn = 0 # Nombre total de true negative
        n_total = 0 # Nombre total de pixels

        for d in dataset:
            label = d['label']
            mask = d['mask']
            image = d['image']

            pred = self.segment_vessels(image)
            label = label[mask]
            pred = pred[mask]
            n = np.sum(d['mask'])

            tp = tp + np.sum(pred & label)
            fp = fp + np.sum(pred & (1-label))
            fn = fn + np.sum(label & (1-pred))
            tn = tn + np.sum((1-label) & (1-pred))
            n_total = n_total + n

        accuracy = (tp + tn)/(tp+tn+fp+fn)
        confusion_matrix = [[tp, fn],[fp, tn]]/n_total

        return accuracy, confusion_matrix, n_total

    def dice(self, dataset: List[dict]) -> float:
        """
        Évalue l'indice Sørensen-Dice de l'algorithme sur un dataset donné et sur la région d'intérêt indiquée par le 
        champ "mask".

        Parameters:
            dataset (List[dict]): Base de données sur laquelle calculer l'indice Dice.

        Returns:
            dice_index (float): Indice de Sørensen-Dice.
        """

        # TODO: II.Q6
        # Vous pouvez utiliser la fonction fournie plus bas : dice().

        dice_index = ...

        return dice_index

    def plot_roc(self, dataset: List[dict]) -> float:
        """
        Affiche la courbe ROC et calcule l'AUR de l'algorithme pour un
        dataset donnée et sur la région d'intérêt indiquée par le champs
        mask.

        Parameters:
            dataset (List[dict]): Base de données sur laquelle calculer l'AUR.

        Returns:
            roc_auc (float): Aire sous la courbe ROC.
        """

        # TODO: II.Q8
        # Utilisez la méthode self.roc(dataset) déjà implémentée.
        fpr, tpr, thresholds = self.roc(dataset)

        plt.plot(fpr, tpr)
        plt.title("ROC curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        roc_auc = auc(fpr, tpr)

        return roc_auc


def load_dataset():
    """Charge les images des ensembles d'entrainement et de test dans 2 listes de dictionnaires. Pour chaque 
    échantillon, il faut créer un dictionnaire dans le dataset contenant les champs ["name", "image", "label", "mask"].
    On pourra ainsi accéder à la premiére image du dataset d"entrainement avec train[0]["image"].
    
    À noter que comme les listes et les dictionnaires sont des objets mutables en Python, la copie intégrale d'un
    dataset peut se faire avec
    >>> from copy import deepcopy
    >>> train_copy = deepcopy(train)
    """
    # On a 20 dictionnaire dans la liste train. Un dictionnaire par image. Le dict contient image, label et mask.

    files = sorted(os.listdir("DRIVE/data/training/"))
    train = []

    for file in files:
        sample = {}
        sample["name"] = file

        # TODO I.Q3 Chargez les images image, label et mask:
        sample["image"] = 1 - imread('DRIVE/data/training/'+file) # Type float, intensité comprises entre 0 et 1. Inverse l'image
        sample["label"] = (imread('DRIVE/label/training/'+file)).astype(bool) # Type booléen
        sample["mask"] = (imread('DRIVE/mask/training/'+file)).astype(bool) # Type booléen

        train.append(sample)

    files_test = sorted(os.listdir("DRIVE/data/test/"))
    test = []

    # TODO I.Q3 De la même manière, chargez les images de test.
    for file in files_test:
        sample = {}
        sample["name"] = file
        sample["image"] = 1 - imread('DRIVE/data/test/' + file)
        sample["label"] = (imread('DRIVE/label/test/' + file)).astype(bool)
        sample["mask"] = (imread('DRIVE/mask/test/' + file)).astype(bool)

        test.append(sample)

    return train, test


def dice(targets, predictions):
    return 2 * np.sum(targets * predictions) / (targets.sum() + predictions.sum())


if __name__=="__main__":
    print("Hello World")

    # Pour fins de debugging : 

    # msld = MSLD(W=15, L=[1, 3, 5, 7, 9, 11, 15], n_orientation=4)
    # print(msld.L)


    # [train, test] = load_dataset()
    #
    # elem = train[0]
    # image0 = elem['image']
    # label0 = elem['label']
    # # plt.imshow(image1)
    # # cv2.imshow("Label 0", label0)
    #
    # # print(image3.max())
    # # print(label3.dtype)
    #
    # R1 = msld.basic_line_detector(image0[:, :, 1], L=1)
    # R15 = msld.basic_line_detector(image0[:, :, 1], L=15)
    #
    # cv2.imshow("L=1", R1)
    # cv2.imshow("L=15", R15)
    #
    #
    # Rcombined = msld.multi_scale_line_detector(image0)
    # cv2.imshow("Rcombined", Rcombined)
    #
    # threshold, accuracy = msld.learn_threshold(train)
    # print(threshold, accuracy)
    #
    # vessels = msld.segment_vessels(image0)
    # # cv2.imshow("Vessels", vessels)
    #
    # test_local = deepcopy(test)
    # kernel = skmorph.disk(3)
    #
    # for d in test_local:
    #     d['mask'] = skmorph.binary_dilation(d['label'], selem=kernel)
    #
    # local_accuracy, local_confusion_matrix, local_total = msld.naive_metrics(test_local)
    #
    # # cv2.waitKey(0)
    #
    # # print(R.shape)
    # # print(R[250, 250, 1])
    # # print(image3.shape)









