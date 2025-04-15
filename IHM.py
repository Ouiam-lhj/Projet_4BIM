from customtkinter import *
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from CTkSpinbox import * 
import io
from code_gen_blend import *
#from autoencoder import *


class ImageDisplayError(Exception):
    pass

class SelectionError(Exception):
    pass

class MethodError(Exception):
    pass
class DynamicGrid():
    
    
    def __init__(self, links, parentFrame, grid_width, grid_height, margin=0.01):
        self.parentFrame = parentFrame
        self.links = links # Modifier links par les vecteurs latents. Si liens, pas nécessaire de modifier
        self.loadImages(links) # Fonction ne marche qu'avec des liens, si nécessaire, rajouter une fonction pour les vecteurs latents
        self.figures = []
        self.width = grid_width
        self.height = grid_height
        self.margin = margin
        self.frames = []
        self.selected_images = []
        self.rows = 0
        self.columns = 0
        self.isEmpty = True
        self.fusionMethod = "BLEND"
        self.nbImage = 7
        self.images_history = []
        self.index_image_history = -1
    
    def figsToImage(self):
        """
        Fonction pour convertir les figures en images PIL.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - Aucune
        """
        for fig in self.figures :
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            img = Image.open(buf)
    
    def getIndexHistory(self):
        """
        Fonction pour obtenir l'index de l'historique des images.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - index_image_history : index de l'historique des images
        """
        return self.index_image_history
    
    def maxIndexHistory(self):
        """
        Fonction pour obtenir le nombre maximum de générations dans l'historique.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - Aucune
        """
        return len(self.images_history)
    # Permet de modifier les figures qui sont utilisée
    def setFigures(self, figures):
        """
        Fonction pour définir les figures à afficher.
        Entrées :
            - self : instance de la classe DynamicGrid
            - figures : liste de figures à afficher
        Sorties :
            - Aucune
        """
        self.figures = figures
    
    def getFigures(self):
        """
        Fonction pour obtenir les figures à afficher.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - figures : liste de figures à afficher
        """

        return self.figures
    
    def setFusionMethod(self, method):
        """
        Fonction pour définir la méthode de fusion.
        Entrées :
            - self : instance de la classe DynamicGrid
            - method : méthode de fusion à utiliser
        Sorties :
            - Aucune
        """
        if method not in ["BLEND", "VAE"]:
            raise ValueError("La méthode de fusion doit être soit 'BLEND' soit 'VAE'")
        self.fusionMethod = method

    def getFusionMethod(self):
        """
        Fonction pour obtenir la méthode de fusion.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - fusionMethod : méthode de fusion utilisée
        """
        return self.fusionMethod

    def setUniqueSelection(self, val):
        """
        Fonction pour définir la sélection unique.
        Entrées :
            - self : instance de la classe DynamicGrid
            - val : valeur de la sélection unique
        Sorties :
            - Aucune
        """
        if isinstance(val, bool):
            raise TypeError
        else:
            self.unique_selection = val
    
    def getUniqueSelection(self):
        """
        Fonction pour obtenir la sélection unique.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - unique_selection : valeur de la sélection unique
        """

        return self.unique_selection
    
    def get_grid_dimensions(self,n):
        """
        Fonction pour obtenir les dimensions de la grille. Assigne le nombre de lignes et de colonnes en fonction du nombre d'images à afficher.
        Entrées :
            - n : nombre d'images à afficher
        Sorties :
            - Aucune
        """
        # Le nombre de lignes sera la partie entière de la racine carrée
        l = int(np.floor(np.sqrt(n)))
        # Pour garantir que rows * columns >= n, on calcule le nombre de colonnes comme le plafond de n / rows
        c = int(np.ceil(n / l))
        self.rows = min(l,c)
        self.columns = max(l,c)

    def getWidth(self):
        """
        Fonction permettant d'obtenir la largeur de la grille.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - width : largeur de la grille
        """
        return self.width

    def getHeight(self):
        """
        Fonction permettant d'obtenir la hauteur de la grille.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - height : hauteur de la grille
        """
        return self.height
    
    def setHeight(self, h):
        """
        Fonction permettant de définir la hauteur de la grille.
        Entrées :
            - self : instance de la classe DynamicGrid
            - h : hauteur de la grille
        Sorties :
            - Aucune
        """
        self.height = h

    def setWidth(self, w):
        """
        Fonction permettant de définir la largeur de la grille.
        Entrées :
            - self : instance de la classe DynamicGrid
            - w : largeur de la grille
        Sorties :
            - Aucune
        """ 
        self.width = w 

    def resizeImages(self, images):
        """
        Fonction pour redimensionner les images à une taille de 250x250 pixels.
        Entrées :
            - self : instance de la classe DynamicGrid
            - images : liste d'images à redimensionner
        Sorties :
            - images : liste d'images redimensionnées"""
        images = list(map(lambda x : x.resize((250, 250), Image.LANCZOS), images))
        return images
    
    def loadImages(self, links):
        """
        Fonction pour charger les images à partir de liens.
        Entrées :
            - self : instance de la classe DynamicGrid
            - links : liste de liens d'images à charger
        Sorties :
            - Aucune
        """
        self.images = list(map(lambda x : Image.open(x), links))

    def addImage(self, image):
        """
        Fonction qui permet d'ajouter une image à la liste d'images.
        Entrées :
            - self : instance de la classe DynamicGrid
            - image : image à ajouter
        Sorties :
            - Aucune
        """
        self.images.append(image)

    def ToCTkImage(self, s, source):
        """
        Fonction pour convertir les images en images CTk.
        Entrées :
            - self : instance de la classe DynamicGrid
            - s : taille des images
            - source : source des images (LINK ou FIGURES)
        Sorties :
            - Aucune
        """

        if (source == "LINK"):
            self.images = list(map(lambda x : CTkImage(light_image=x, dark_image=x, size = (s,s)),self.images))
        elif (source == "FIGURES"):
            self.images = list(map(lambda x : CTkImage(light_image=x, dark_image=x, size = (s,s)),self.figures))
        else:
            raise KeyError("source ne peut prendre que deux valeurs : LINK ou FIGURES")
    
    def getImages(self):
        """
        Fonction pour obtenir les images.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - images : liste d'images
        """
        return self.images
    
    def getImage(self, k):
        """
        Fonction pour obtenir une image à un index donné.
        Entrées :
            - self : instance de la classe DynamicGrid
            - k : index de l'image à obtenir
        Sorties :
            - image : image à l'index k
        """
        if k > len(self.images):
            raise IndexError
        return self.images[k]
    
    def setNbImage(self, n):
        """
        Fonction pour définir le nombre d'images de la grille.
        Entrées :
            - self : instance de la classe DynamicGrid
            - n : nombre d'images à afficher
        Sorties :
            - Aucune
        """
        if n < 0:
            raise ValueError("Le nombre d'images doit être supérieur à 0")
        self.nbImage = n
        self.get_grid_dimensions(n)


    def algoGen(self):
        """
        Fonction pour exécuter l'algorithme génétique et générer de nouvelles images.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - Aucune
        """

        # Fonction pour tout tester pour le moment.
        # On récupère les images sélectionnées
        if (len(self.selected_images) < 2):
            raise SelectionError("Vous n'avez pas sélectionné assez d'images")
        
        pil_images = list(map(lambda x : x.cget("light_image"), self.selected_images))
        
        # On les passes à l'algo génétique
        if self.fusionMethod == "BLEND":
            #On prend simplement les deux premières images
            arrays = apply_random_blending(pil_images,self.nbImage)
            self.figures = list(map(lambda x: Image.fromarray(x), arrays))
            self.figures = self.resizeImages(self.figures)
            print(self.figures[0].size)
            print("FIGURES : {}".format(self.figures))
        # On a obtenue les (ou la dans le cadre du premier test)
        elif (self.fusionMethod == "VAE"):
            print("En cours d'implémentation")
            return
            #pil_images = vae_generate_mutated_images(var_encoder, var_decoder, self.selected_images, new_to_show=self.nbImage, mutation_strength=0.5)
            #self.figures = self.resizeImages(self.figures)
        else:
            raise MethodError("Un mot clef incorrect a été utilisé pour le passage de génération")
    
        self.destroyGrid()
        self.displayImages(source="FIGURES", add_to_history=True)
        self.selected_images = []


    def displayImages(self, source = "LINK", add_to_history = False):
        """
        Fonction pour afficher les images dans la grille.
        Entrées :
            - self : instance de la classe DynamicGrid
            - source : source des images (LINK ou FIGURES)
            - add_to_history : booléen pour ajouter les images à l'historique
        Sorties :
            - Aucune
        """
        
        if (self.isEmpty == False):
            self.destroyGrid()

        images_displayed=0

        nb_images = self.nbImage
        self.get_grid_dimensions(nb_images)

        width_frame = self.width/self.rows *0.7
        height_frame = self.height/self.columns *0.7
        
        images_size = max(width_frame/self.rows , height_frame/self.columns)

        if source == "LINK":
            self.loadImages(self.links)
            self.ToCTkImage(images_size,source)
        elif source == "FIGURES":
            self.ToCTkImage(images_size,source)
        elif source == "IMAGE":
            pass
        else:
            return

        if (self.images == []):
            raise ImageDisplayError

        print("Image à afficher : {}".format(self.images))
        for i in range(self.columns):
            currentFrame = CTkFrame(self.parentFrame, width=width_frame, height=height_frame, fg_color="transparent")
            currentFrame.pack(expand=True, fill="both", side="left")
            self.frames.append(currentFrame)
            for j in range(images_displayed, images_displayed + self.rows):
                button = CTkButton(currentFrame, image= self.images[j],width=images_size, height=images_size, fg_color="transparent", hover=True, hover_color="#000000",text='')
                button.pack(side = TOP)
                button.configure(command = (lambda button=button, img=self.images[j] : self.changeButtonColor(button, img)))
            images_displayed += self.rows
        self.isEmpty = False

        if add_to_history == True:
            if (self.index_image_history == len(self.images_history) - 1):
                self.images_history.append([self.images.copy(), self.selected_images.copy()])
            else:
                self.images_history = self.cutCurrentHistory()
                self.images_history.append([self.images.copy(), self.selected_images.copy()])
            self.images = []
            self.selected_images = []
            self.index_image_history = len(self.images_history) - 1
    
    def cutCurrentHistory(self):
        """
        Fonction pour couper l'historique actuel des images.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - new_history : nouvelle liste d'historique des images
        """
        if self.index_image_history == -1:
            return self.images_history
        new_history = []
        for i in range(self.index_image_history+1):
            new_history.append(self.images_history[i])
        return new_history
    
    def previousImages(self):
        """
        Fonction pour charger les images de la génération précédente.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - Aucune
        """
        if self.index_image_history <= 0:
            raise ValueError("Impossible de charger des images avant la première génération")
        self.images = self.images_history[self.index_image_history - 1][0]
        self.displayImages(source="IMAGE", add_to_history=False)
        self.index_image_history -= 1
    
    def nextImages(self):
        """
        Fonction pour charger les images de la génération suivante.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - Aucune
        """

        if (self.index_image_history == (len(self.images_history) - 1) or self.index_image_history == -1):
            raise ValueError("Impossible de charger des images qui n'ont pas encore été générée")
        self.images = self.images_history[self.index_image_history + 1][0]
        self.displayImages(source="IMAGE", add_to_history=False)
        self.index_image_history += 1

    def resetImages(self):
        """
        Fonction pour réinitialiser les images à la génération initiale.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - Aucune
        """
        self.selected_images = self.images_history[self.index_image_history - 1][1]
        self.algoGen()
        self.index_image_history += 1

    def destroyGrid(self):
        """
        Fonction pour détruire la grille d'images.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - Aucune
        """

        for frame in self.frames:
            frame.destroy()
        self.frames = []
        self.isEmpty = True
        
    
    def changeButtonColor(self, button, image):
        """
        Fonction pour changer la couleur d'un bouton et gérer la sélection d'images.
        Entrées :
            - button : le bouton à modifier
            - image : l'image associée au bouton
        Sorties :
            - Aucune
        """
        current_color = button.cget("fg_color")
        new_color = "#000000" if current_color == "transparent" else "transparent"
        button.configure(fg_color=new_color)

        if image not in self.selected_images:
            self.selected_images.append(image)
        else:
            if image in self.selected_images:
                self.selected_images.remove(image)
    
    def get_selected_images(self):
        """
        Fonction pour obtenir les images sélectionnées.
        Entrées :
            - self : instance de la classe DynamicGrid
        Sorties :
            - selected_images : liste des images sélectionnées
        """
        print(self.selected_images)
        return self.selected_images
    
class IHM():

    def __init__(self):
        self.root = CTk()
        self.homePage()
        set_appearance_mode("light")
        self.grid.nbImage = 6
        self.root.mainloop()

    def homePage(self):
        """
        Fonction d'affichage de la page d'accueil de l'application.
        Entrées :
            - self : instance de la classe IHM
        Sorties :
            - Aucune
        """
        self.root.title("Le profiler des zencoders")
        self.root.geometry("960x590")
        self.root.resizable(False, False)

        self.principalMainframe = CTkFrame(self.root, fg_color="#ffffff", border_width = 0)
        self.menuMainframe = CTkFrame(self.root, width= 200)
        self.titleFrame = CTkFrame(self.principalMainframe, fg_color="#ffffff", height = 70)
        self.photosFrame = CTkFrame(self.principalMainframe, fg_color="#ffffff")

        self.buttonsFrame=CTkFrame(self.principalMainframe, fg_color="#ffffff", height = 50)
        self.leftSideButtonFrame = CTkFrame(self.buttonsFrame, fg_color="#ffffff", height = 50)
        self.middleSideButtonFrame = CTkFrame(self.buttonsFrame, fg_color="#ffffff", height = 50)
        self.rightSideButtonFrame = CTkFrame(self.buttonsFrame, fg_color="#ffffff", height = 50)
        self.consignes_label = CTkLabel(self.photosFrame, text="Bienvenue\nVeuillez remplir le formulaire pour commencer", font=("Arial", 30), text_color="#38393b")

        self.menuFormButton = CTkButton(self.menuMainframe, text="Formulaire", command=self.displayFormulaire, fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'], text_color='#333333')
        self.menuExportButton = CTkButton(self.menuMainframe, text="Exporter", command=self.displayExportWindow, fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'], text_color='#333333')
        self.menuParamButton = CTkButton(self.menuMainframe, text="Paramètres", command=self.displayParameterWindow, fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'], text_color='#333333')
        self.grid = DynamicGrid([], self.photosFrame, self.photosFrame.winfo_width() ,self.photosFrame.winfo_height())
        self.newGenButton = CTkButton(self.middleSideButtonFrame, width=100, height=35, text = "Nouvelle génération", command = lambda : self.grid.algoGen(), fg_color="transparent", hover_color=['#e4e4eb', '#3a3b3d'], text_color='#333333')
        self.previousGenButton = CTkButton(self.middleSideButtonFrame, width=50, height=35, text = "←", command = lambda : self.previousGen(), fg_color="transparent", hover_color=['#e4e4eb', '#3a3b3d'], text_color='#333333',state="disable")
        self.nextGenButton = CTkButton(self.middleSideButtonFrame, width=50, height=35, text = "→", command = lambda : self.nextGen(), fg_color="transparent", hover_color=['#e4e4eb', '#3a3b3d'], text_color='#333333', state="disable")
        
        self.principalMainframe.pack(expand=True, fill="both", side="right")
        self.menuMainframe.pack(fill="y", side="left")
        self.titleFrame.pack(expand=True, fill="both", side="top")
        self.photosFrame.pack(expand=True, fill="both", side="top")
        self.buttonsFrame.pack(expand=True, fill="both", side="top")
        self.leftSideButtonFrame.pack(expand=True, fill="x", side="left")
        self.middleSideButtonFrame.pack(expand=True, fill="x", side="left")
        self.rightSideButtonFrame.pack(expand=True, fill="x", side="left")
        self.previousGenButton.pack(side="left")
        self.newGenButton.pack(side="left")
        self.nextGenButton.pack(side="left")
        self.consignes_label.pack(expand=True, fill="both", side="top", pady=200)

        self.menuFormButton.pack(fill="x", pady=10)
        self.menuExportButton.pack(fill="x", pady=10)
        self.menuParamButton.pack(fill="x", pady=10)

    def displayFormulaire(self):
        """
        Fonction d'affichage du formulaire de pré-sélection.
        Entrées :
            - self : instance de la classe IHM
        Sorties :
            - Aucune
        """
        app = CTkToplevel(self.root)
        app.title("Questionnaire")
        app.geometry("520x600")
        app.configure(bg="#f5f5f5")
        app.grid_rowconfigure(0, weight=1)
        app.grid_columnconfigure(0, weight=1)

        scroll = CTkScrollableFrame(app, width=500, height=600, fg_color="#ffffff")
        scroll.grid(row=0, column=0, columnspan=3, sticky="nsew")
        scroll.grid_rowconfigure(0, weight=1)
        scroll.grid_columnconfigure(0, weight=1)
        app.grab_set()
        
        titre = "Questionnaire de pré-sélection"
        consignes = "Veuillez répondre aux questions suivantes pour l'identification de l'individu. Si vous avez le moindre doute, cochez la case \"Je ne sais pas\".\n\nN'oubliez pas de sauvegarder vos réponses en cliquant sur le bouton \"Valider\" en bas de page."

        label_titre = CTkLabel(scroll, text=titre, font=("Arial", 30), wraplength=480, text_color="#333333")
        label_consignes = CTkLabel(scroll, text=consignes, font=("Arial", 18), wraplength=480, text_color="#555555")
        separator = CTkFrame(scroll, height=2, fg_color="#cccccc")

        text_color = "#333333"
        font_size_questions = 16
        text_sex = CTkLabel(scroll, text="Quel était le sexe de l'individu ?", font=("Arial", font_size_questions), text_color=text_color)
        text_cheveux = CTkLabel(scroll, text="Quelle était la couleur des cheveux de l'individu ?", font=("Arial", font_size_questions), text_color=text_color)
        text_texture = CTkLabel(scroll, text="Quel était la texture de cheveux de l'individu ?", font=("Arial", font_size_questions), text_color=text_color)
        text_age = CTkLabel(scroll, text="L'individu vous paraissait plutôt :", font=("Arial", font_size_questions), text_color=text_color)

        self.combobox_sex = CTkComboBox(scroll, values=["Homme", "Femme", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_sex.set("Sélectionner")

        self.combobox_cheveux = CTkComboBox(scroll, values=["Noirs", "Bruns/Châtains", "Blonds", "Gris", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_cheveux.set("Sélectionner")

        self.combobox_texture = CTkComboBox(scroll, values=["Lisses", "Bouclés", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_texture.set("Sélectionner")

        self.combobox_age = CTkComboBox(scroll, values=["Jeune", "Agé", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_age.set("Sélectionner")

        button = CTkButton(scroll, text="Valider", font=("Arial", font_size_questions), width=200, height=30, command=lambda: self.close_window(app), fg_color="#528868", text_color="#ffffff")

        label_titre.grid(row=0, column=0, columnspan=3, pady=(20, 10))
        label_consignes.grid(row=1, column=0, columnspan=3, pady=(10, 20))
        separator.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 20))

        text_sex.grid(row=3, column=0, padx=20, pady=(20, 2))
        self.combobox_sex.grid(row=4, column=0, padx=20, pady=(2, 20))

        text_cheveux.grid(row=7, column=0, padx=20, pady=(20, 2))
        self.combobox_cheveux.grid(row=8, column=0, padx=20, pady=(2, 20))

        text_texture.grid(row=9, column=0, padx=20, pady=(20, 2))
        self.combobox_texture.grid(row=10, column=0, padx=20, pady=(2, 20))

        text_age.grid(row=15, column=0, padx=20, pady=(20, 2))
        self.combobox_age.grid(row=16, column=0, padx=20, pady=(2, 20))

        button.grid(row=19, column=0, pady=(50, 20))

    def close_window(self, app):
        """
        Fonction de fermeture de la fenêtre du formulaire. Permet de sélectionner aléatoirement des images ayant les critères recherchés.
        Entrées :
            - app : instance de la fenêtre du formulaire
        Sorties :
            - Aucune
        """
        reponses = {
            "Male": self.combobox_sex.get(),
            "Brown_Hair": self.combobox_cheveux.get(),
            "Gray_Hair": self.combobox_cheveux.get(),
            "Black_Hair": self.combobox_cheveux.get(),
            "Blond_Hair": self.combobox_cheveux.get(),
            "Wavy_Hair": self.combobox_texture.get(),
            "Straight_Hair": self.combobox_texture.get(),
            "Young": self.combobox_age.get(),
        }

        if "Sélectionner" in reponses.values():
            messagebox.showerror("", "Veuillez répondre à toutes les questions.")
        else:
            converted_reponses = self.conversion_reponses(reponses)
            messagebox.showinfo("", "Vos réponses ont bien été enregistrées.")    
        try:
            sample = self.chose_first_sample_photo(self.get_photos_matching_form(converted_reponses))
            list_sample = sample.tolist()
            app.destroy()
            list_sample = [f"img_align_celeba/{item}" for item in list_sample]
            self.grid.links = list_sample
            self.displayGrid()
        except ValueError:
            messagebox.showerror("Erreur dans la sélection", "Les critères sont trop spécifiques. Impossible de générer assez d'images. Veuillez recommencer.")

    def conversion_reponses(self, reponses):
        """
        Fonction de conversion des réponses du formulaire en valeurs numériques afin de pouvoir les utiliser pour trier le dataframe.
        Entrées :
            - reponses : dictionnaire contenant les réponses du formulaire
        Sorties :
            - reponses : dictionnaire contenant les réponses converties
        """
        conversions = {
        "Male": {"Homme": 1, "Femme": -1, "Je ne sais pas": 0},
        "Brown_Hair": {"Noirs": -1, "Bruns/Châtains": 1, "Blonds": -1, "Gris": -1, "Je ne sais pas": 0},
        "Gray_Hair": {"Noirs": -1, "Bruns/Châtains": -1, "Blonds": -1, "Gris": 1, "Je ne sais pas": 0},
        "Black_Hair": {"Noirs": 1, "Bruns/Châtains": -1, "Blonds": -1, "Gris": -1, "Je ne sais pas": 0},
        "Blond_Hair": {"Noirs": -1, "Bruns/Châtains": -1, "Blonds": 1, "Gris": -1, "Je ne sais pas": 0},
        "Wavy_Hair": {"Lisses": -1, "Bouclés": 1, "Je ne sais pas": 0},
        "Straight_Hair": {"Lisses": 1, "Bouclés": -1, "Je ne sais pas": 0},
        "Young": {"Jeune": 1, "Agé": -1, "Je ne sais pas": 0},
        }
        
        for key in reponses:
            if key in conversions and reponses[key] in conversions[key]:
                reponses[key] = conversions[key][reponses[key]]
            else:
                reponses[key] = 0 

        return reponses

    def get_photos_matching_form(self, reponses):
        """
        Fonction pour obtenir les photos correspondant aux réponses du formulaire.
        Entrées :
            - reponses : dictionnaire contenant les réponses du formulaire
        Sorties :
            - df_form : dataframe contenant les photos correspondant aux réponses
        """
        dropping = []
        df_attr = pd.read_csv("final_dataset.txt" , sep = "\s+", header=0)

        for key, value in reponses.items():
            if value == 0:
                continue

            for index, val in zip(df_attr[key].index, df_attr[key].values):
                if val != value:  # Correction de 'ind' qui était une variable inexistante
                    dropping.append(index)
        df_form = df_attr.drop(dropping)

        return df_form
    
    def chose_first_sample_photo(self, df_form):
        """
        Fonction pour choisir un échantillon aléatoire de photos à partir du dataframe.
        Entrées :
            - df_form : dataframe contenant les photos correspondant aux réponses
        Sorties :
            - sample : échantillon aléatoire de photos
        """

        sample = df_form.sample(n=6)
        return sample.index


    def displayGrid(self):
        """
        Fonction d'affichage de la grille d'images.
        Entrées :
            - self : instance de la classe IHM
        Sorties :
            - Aucune
        """
        if self.consignes_label.winfo_exists():
            self.consignes_label.pack_forget()
        print(self.root.winfo_geometry())
        self.grid.setHeight(self.photosFrame.winfo_height())
        self.grid.setWidth(self.photosFrame.winfo_width())


        self.grid.displayImages(add_to_history=True)
    
    def nextGen(self):
        """
        Fonction de gestion de l'affichage de la génération suivante.
        Entrées :
            - self : instance de la classe IHM
        Sorties :
            - Aucune
        """
        self.grid.nextImages()
        self.updateButtonStatus()
    
    def previousGen(self):
        """
        Fonction de gestion de l'affichage de la génération précédente.
        Entrées :
            - self : instance de la classe IHM
        Sorties :
            - Aucune
        """
        self.grid.previousImages()
        self.updateButtonStatus()


    def displayParameterWindow(self):
        """
        Fonction d'affichage de la fenêtre de paramètres.
        Entrées :
            - self : instance de la classe IHM
        Sorties :
            - Aucune
        """
        if hasattr(self, 'disp_window') and self.disp_window.winfo_exists():
            self.disp_window.focus_force()
            return
        
        self.disp_window = None
        self.disp_window = CTkToplevel(self.root)
        self.disp_window.title("Paramètres")
        self.disp_window.geometry("360x240")
        self.disp_window.resizable(False, False)

        topFrame = CTkFrame(self.disp_window)
        midFrame = CTkFrame(self.disp_window)
        bottomFrame = CTkFrame(self.disp_window)

        topFrame.pack(pady=10, fill="x")
        midFrame.pack(padx=20, pady=10, fill="both", expand=True)
        bottomFrame.pack(pady=10)

        title = CTkLabel(topFrame, text="Paramètres", font=("Arial", 24, "bold"))
        title.pack()

        self.temp_nbGenImages = IntVar(value=self.grid.nbImage)
        self.temp_checkVarFus = StringVar(value=self.grid.fusionMethod)

        textImage = CTkLabel(midFrame, text="Images par génération :")
        spinboxImages = CTkSpinbox(midFrame, variable=self.temp_nbGenImages, min_value=4, max_value=9,
                                width=80, height=28, border_width=1)
        spinboxImages.set(self.grid.nbImage)

        textFus = CTkLabel(midFrame, text="Méthode de fusion :")
        comboFus = CTkComboBox(midFrame, values=['BLEND', 'VAE'], variable=self.temp_checkVarFus)


        textImage.grid(row=1, column=0, sticky="w", pady=5)
        spinboxImages.grid(row=1, column=1, sticky="e", pady=5)
        textFus.grid(row=2, column=0, sticky="w", pady=5)
        comboFus.grid(row=2, column=1, sticky="e", pady=5)

        validateButton = CTkButton(bottomFrame, text="Sauvegarder", command=self.saveParameters)
        cancelButton = CTkButton(bottomFrame, text="Annuler", fg_color="grey", hover_color="#999",
                                command=self.confirmCloseWithoutSave)

        validateButton.grid(row=0, column=0, padx=10)
        cancelButton.grid(row=0, column=1, padx=10)

        self.disp_window.protocol("WM_DELETE_WINDOW", self.confirmCloseWithoutSave)
        self.disp_window.grab_set()
        self.disp_window.focus_force()


    def confirmCloseWithoutSave(self):
        """
        Fonction de confirmation de la fermeture de la fenêtre sans sauvegarder les paramètres.

        Entrées :
            - self : instance de la classe IHM
        Sorties :
            - Aucune
        """
        response = messagebox.askyesno("Confirmation", "Souhaitez-vous quitter sans sauvegarder ?")
        if response:
            self.disp_window.grab_release()
            self.disp_window.destroy()
        else:
            self.disp_window.focus_force()


    def saveParameters(self):
        """
        Fonction de sauvegarde des paramètres.

        Entrées :
            - self : instance de la classe IHM
        Sorties :
            - Aucune
        """
        # self.grid.nbImage = self.temp_nbGenImages.get() - On garde fixé à 6 pour le moment.
        self.grid.fusionMethod = self.temp_checkVarFus.get()
        self.disp_window.grab_release()
        self.disp_window.destroy()
    
    def displayExportWindow(self):
        """
        Fonction d'affichage de la fenêtre d'exportation.
        Entrées :
            - self : instance de la classe IHM
        Sorties :
            - Aucune
        """
        if hasattr(self, 'export_window') and self.export_window.winfo_exists():
            self.export_window.focus_force()
            return
        self.export_window = None
        self.export_window = CTkToplevel(self.root)
        self.export_window.title("Export")
        self.export_window.geometry("400x250")

        label_folder = CTkLabel(self.export_window, text="Choisissez un dossier d'exportation")
        self.folder_path_var = StringVar(value="")
        self.folder_entry = CTkEntry(self.export_window, textvariable=self.folder_path_var, width=300)
        folder_btn = CTkButton(self.export_window, text="Sélectionner un dossier", command=self.select_folder)
        
        label_format = CTkLabel(self.export_window, text="Choisissez un format de sortie")   
        self.format_options = ["PNG", "JPG"]
        self.format_var = StringVar(value="PNG")
        self.format_menu = CTkOptionMenu(self.export_window, variable=self.format_var, values=self.format_options)

        label_folder.pack(pady=(20, 10))
        self.folder_entry.pack()
        folder_btn.pack(pady=(10, 10))
        label_format.pack(pady=(10, 5))
        self.format_menu.pack()

        export_btn = CTkButton(self.export_window, text="Exportation Image", command=self.export_images)
        export_btn.pack(pady=(20, 10))



    def select_folder(self):
        """
        Fonction pour sélectionner un dossier d'exportation.
        Entrées :
            - self : instance de la classe IHM
        Sorties :
            - Aucune
        """
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path_var.set(folder)
        self.export_window.grab_set()
        self.export_window.focus_force()
        


    def export_images(self):
        """
        Fonction pour exporter les images sélectionnées dans le format choisi.
        Entrées :
            - self : instance de la classe IHM
        Sorties :
            - Aucune
        """
        if not hasattr(self, 'grid'):
            messagebox.showerror("Erreur", "Aucune image à exporter !")
            self.export_window.focus_force()
            return
        if not hasattr(self, 'folder_path_var'):
            messagebox.showerror("Erreur", "Aucun dossier sélectionné !")
            self.export_window.focus_force()
            return
        folder = self.folder_path_var.get()
        if not folder:
            messagebox.showerror("Erreur", "Sélectionnez un dossier pour l'export !")
            self.export_window.focus_force()
            return
        
        export_format = self.format_var.get()

        if not self.grid.selected_images :
            messagebox.showerror("Erreur", "Pas d'images sélectionnée pour l'export !")
            self.export_window.focus_force()
            return


        count = 1
        pil_images = list(map(lambda x : x.cget("light_image"), self.grid.selected_images))
        for pil_img in pil_images:

            if export_format == "JPG":
                if pil_img.mode in ("RGBA", "P"):
                    pil_img = pil_img.convert("RGB")
            file_path = os.path.join(folder, f"test_image_{count}.{export_format.lower()}")

            try:
                pil_img.save(file_path)

            except Exception as e:
                messagebox.showerror("Erreur d'exportation", f"Impossible de sauvegarder l'image {count} : {str(e)}")
                return
            count += 1

        messagebox.showinfo("Export", "Les images ont été exportées avec succès !")
        self.export_window.destroy()
        

if __name__ == "__main__":
    test = IHM()