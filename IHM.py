from customtkinter import *
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from CTkSpinbox import * 
import io
from code_gen_blend import *

#pip install customtkinter
#pip install CTkListbox
#pip install tkinter
#pip install PIL
#pip install CTkSpinbox

class ImageDisplayError(Exception):
    pass

class SelectionError(Exception):
    pass

class MethodError(Exception):
    pass
class DynamicGrid():
    
    
    def __init__(self, links, parentFrame, grid_width, grid_height, margin=0.01, unique_selection = False):
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
        self.unique_selection = unique_selection
        self.fusionMethod = "BLEND"
    
    def figsToImage(self):
        for fig in self.figures :
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            img = Image.open(buf)
    
    # Permet de modifier les figures qui sont utilisée
    def setFigures(self, figures):
        self.figures = figures
    
    def getFigures(self, figures):
        return figures
    
    def setFusionMethod(self, method):
        self.fusionMethod = method

    def getFusionMethod(self):
        return self.fusionMethod

    def setUniqueSelection(self, val):
        if isinstance(val, bool):
            raise TypeError
        else:
            self.unique_selection = val
    
    def getUniqueSelection(self):
        return self.unique_selection
    
    def get_grid_dimensions(self,n):
        # Le nombre de lignes sera la partie entière de la racine carrée
        l = int(np.floor(np.sqrt(n)))
        # Pour garantir que rows * columns >= n, on calcule le nombre de colonnes comme le plafond de n / rows
        c = int(np.ceil(n / l))
        self.rows = min(l,c)
        self.columns = max(l,c)

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height
    
    def setHeight(self, h):
        self.height = h

    def setWidth(self, w):
        self.width = w 

    def loadImages(self, links):
        print("chargement des images..")
        self.images = list(map(lambda x : Image.open(x), links))
        print(self.images)
        print("Les images sont chargées !")

    def addImage(self, image):
        self.images.append(image)

    def ToCTkImage(self, s, source):
        if (source == "LINK"):
            self.images = list(map(lambda x : CTkImage(light_image=x, dark_image=x, size = (s,s)),self.images))
        elif (source == "FIGURES"):
            self.images = list(map(lambda x : CTkImage(light_image=x, dark_image=x, size = (s,s)),self.figures))
        else:
            raise KeyError("source ne peut prendre que deux valeurs : LINK ou FIGURES")
    
    def getImages(self):
        return self.images
    
    def getImage(self, k):
        if k > len(self.images):
            raise IndexError
        return self.images[k]

    def algoGen(self):
        # Fonction pour tout tester pour le moment.
        # On récupère les images sélectionnées
        if (len(self.selected_images) < 2):
            raise SelectionError("Vous n'avez pas sélectionné assez d'images")
        
        pil_images = list(map(lambda x : x.cget("light_image"), self.selected_images))
        
        # On les passes à l'algo génétique
        if self.fusionMethod == "BLEND":
            #On prend simplement les deux premières images
            blended_face = blend_faces(pil_images[0], pil_images[1], alpha=0.45)
            new_face = blended_face[0]
            new_face_attr = blended_face[1]
            new_face_floue = apply_blur_around_face(new_face, new_face_attr, blur_strength=11)
            
            print(type(new_face_floue))

        # On a obtenue les (ou la dans le cadre du premier test)
        elif (self.fusionMethod == "VAE"):
            print("Il me faut les mutations pour réaliser la fusions des codes")
        else:
            raise MethodError("Un mot clef incorrect a été utilisé pour le passage de génération")
        
        self.figures = list(map(lambda x : Image.fromarray(x), [new_face_floue for i in range(6)]))
        self.destroyGrid()
        self.displayImages(source="FIGURES")


        # Le résultat est stocké dans une variable et on display les images.
    def displayImages(self, source = "LINK"):
        # La frame doit être un objet CTkFrame
        # images est une liste d'images
        if (self.images == []):
            raise ImageDisplayError
        
        if (self.isEmpty == False):
            self.destroyGrid()

        images_displayed=0

        print(self.images)
        nb_images = len(self.images)
        self.get_grid_dimensions(nb_images)

        width_frame = self.width/self.rows
        height_frame = self.height/self.columns
        
        images_size = max(width_frame/self.rows , height_frame/self.columns)
        if source == "LINK":
            self.loadImages(self.links)
            self.ToCTkImage(images_size,source)
        elif source == "FIGURES":
            self.ToCTkImage(images_size,source)
        else:
            #Si la grille se supprime sans rien afficher, alors le keyword est le pb
            return

        print(self.images)
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
    
    def destroyGrid(self):
        print("destroyGrid has been called !")
        for frame in self.frames:
            print(frame)
            frame.destroy()
        self.isEmpty = True
        
    
    def changeButtonColor(self, button, image):
        current_color = button.cget("fg_color")
        new_color = "#000000" if current_color == "transparent" else "transparent"
        button.configure(fg_color=new_color)

        if image not in self.selected_images:
            self.selected_images.append(image)
        else:
            if image in self.selected_images:
                self.selected_images.remove(image)
    
    def get_selected_images(self):
        print(self.selected_images)
        return self.selected_images
    
class IHM():

    def __init__(self):
        self.root = CTk()
        self.homePage()
        set_appearance_mode("light")
        self.root.mainloop()

    def homePage(self):
        self.images = ["000001.jpg","000002.jpg","000003.jpg","000004.jpg","000005.jpg","000006.jpg"]

        self.root.title("Le profiler des zencoders")

        self.principalMainframe = CTkFrame(self.root, fg_color="#38393b", border_width = 0)
        self.menuMainframe = CTkFrame(self.root, width= 200)
        self.titleFrame = CTkFrame(self.principalMainframe, fg_color="#00FF00", height = 70)
        self.photosFrame = CTkFrame(self.principalMainframe, height = 500)

        self.buttonsFrame=CTkFrame(self.principalMainframe, fg_color="#FF0000", height = 50)
        self.leftSideButtonFrame = CTkFrame(self.buttonsFrame, fg_color="#FFC0CB", height = 50)
        self.middleSideButtonFrame = CTkFrame(self.buttonsFrame, fg_color="#FFFF00", height = 50)
        self.rightSideButtonFrame = CTkFrame(self.buttonsFrame, fg_color="#FF4500", height = 50)
        self.consignes_label = CTkLabel(self.photosFrame, text="Bienvenue\nVeuillez remplir le formulaire pour commencer", font=("Arial", 30), text_color="#38393b")

        self.menuFormButton = CTkButton(self.menuMainframe, text="Formulaire", command=self.displayFormulaire, fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'])
        self.menuExportButton = CTkButton(self.menuMainframe, text="Exporter", command=self.displayExportWindow, fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'])
        self.menuParamButton = CTkButton(self.menuMainframe, text="Paramètres", command=self.displayParameterWindow, fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'])
        self.photo = CTkButton(self.menuMainframe, text="Test", command= lambda : self.displayGrid(), fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'])
        self.grid = DynamicGrid(self.images, self.photosFrame, self.photosFrame.winfo_width() ,self.photosFrame.winfo_height())
        self.newGenButton = CTkButton(self.middleSideButtonFrame, text = "Nouvelle génération", command = lambda : self.grid.algoGen(), fg_color="transparent", hover_color=['#e4e4eb', '#3a3b3d'])
        self.previousGenButton = CTkButton(self.middleSideButtonFrame, text = "Génération précédente", command = lambda : self.nextGen(), fg_color="transparent", hover_color=['#e4e4eb', '#3a3b3d'])
        self.nextGenButton = CTkButton(self.middleSideButtonFrame, text = "Génération Suivante", command = lambda : self.previousGen(), fg_color="transparent", hover_color=['#e4e4eb', '#3a3b3d'])
        
        self.principalMainframe.pack(expand=True, fill="both", side="right")
        self.menuMainframe.pack(fill="y", side="left")
        self.titleFrame.pack(expand=True, fill="both", side="top")
        self.photosFrame.pack(expand=True, fill="both", side="top")
        self.buttonsFrame.pack(expand=True, fill="both", side="top")
        self.leftSideButtonFrame.pack(expand=True, fill="both", side="left")
        self.middleSideButtonFrame.pack(expand=True, fill="both", side="left")
        self.rightSideButtonFrame.pack(expand=True, fill="both", side="left")
        self.previousGenButton.pack(side="left")
        self.newGenButton.pack(side="left")
        self.nextGenButton.pack(side="left")
        self.consignes_label.pack(expand=True, side="top", pady=50)

        self.menuFormButton.pack(fill="x", pady=10)
        self.menuExportButton.pack(fill="x", pady=10)
        self.menuParamButton.pack(fill="x", pady=10)
        self.photo.pack(fill="x", pady=15)


    def displayFormulaire(self):
        app = CTkToplevel(self.root)
        app.title("Questionnaire")
        app.geometry("520x600")
        app.configure(bg="#f5f5f5")
        app.grid_rowconfigure(0, weight=1)
        app.grid_columnconfigure(0, weight=1)
        app.grab_set()

        # Scrollable frame
        scroll = CTkScrollableFrame(app, width=500, height=600, fg_color="#ffffff")
        scroll.grid(row=0, column=0, columnspan=3, sticky="nsew")
        scroll.grid_rowconfigure(0, weight=1)
        scroll.grid_columnconfigure(0, weight=1)
        
        # Définition des variables
        titre = "Questionnaire de pré-sélection"
        consignes = "Veuillez répondre aux questions suivantes pour l'identification de l'individu. Si vous avez le moindre doute, cochez la case \"Je ne sais pas\".\n\nN'oubliez pas de sauvegarder vos réponses en cliquant sur le bouton \"Valider\" en bas de page."

        # Titres
        label_titre = CTkLabel(scroll, text=titre, font=("Arial", 30), wraplength=480, text_color="#333333")
        label_consignes = CTkLabel(scroll, text=consignes, font=("Arial", 18), wraplength=480, text_color="#555555")
        separator = CTkFrame(scroll, height=2, fg_color="#cccccc")

        # Questions
        text_color = "#333333"
        font_size_questions = 16
        text_sex = CTkLabel(scroll, text="Quel était le sexe de l'individu ?", font=("Arial", font_size_questions), text_color=text_color)
        text_bald = CTkLabel(scroll, text="L'individu était-il chauve ?", font=("Arial", font_size_questions), text_color=text_color)
        text_cheveux = CTkLabel(scroll, text="Quelle était la couleur des cheveux/de la barbe de l'individu ?", font=("Arial", font_size_questions), text_color=text_color)
        text_texture = CTkLabel(scroll, text="Quel était la texture de cheveux de l'individu ?", font=("Arial", font_size_questions), text_color=text_color)
        text_corpu = CTkLabel(scroll, text="Quelle était la corpulence de l'individu ?", font=("Arial", font_size_questions), text_color=text_color)
        text_lunettes = CTkLabel(scroll, text="L'individu portait-il des lunettes ?", font=("Arial", font_size_questions), text_color=text_color)
        text_age = CTkLabel(scroll, text="L'individu vous paraissait plutôt :", font=("Arial", font_size_questions), text_color=text_color)
        text_pilosité = CTkLabel(scroll, text="L'individu portait-il une barbe, une moustache ou un bouc ?", font=("Arial", font_size_questions), text_color=text_color)


        # Réponses (Création et set)
        self.combobox_sex = CTkComboBox(scroll, values=["Homme", "Femme", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_sex.set("Sélectionner")

        self.combobox_bald = CTkComboBox(scroll, values=["Oui", "Non", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_bald.set("Sélectionner")

        self.combobox_cheveux = CTkComboBox(scroll, values=["Noirs", "Bruns/Châtains", "Blonds", "Gris", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_cheveux.set("Sélectionner")

        self.combobox_texture = CTkComboBox(scroll, values=["Lisses", "Bouclés", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_texture.set("Sélectionner")

        self.combobox_corpu = CTkComboBox(scroll, values=["Faible", "Forte", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_corpu.set("Sélectionner")

        self.combobox_lunettes = CTkComboBox(scroll, values=["Oui", "Non", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_lunettes.set("Sélectionner")

        self.combobox_age = CTkComboBox(scroll, values=["Jeune", "Agé", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_age.set("Sélectionner")

        self.combobox_pilosité = CTkComboBox(scroll, values=["Oui", "Non", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_pilosité.set("Sélectionner")

        # Bouton de validation
        button = CTkButton(scroll, text="Valider", font=("Arial", font_size_questions), width=200, height=30, command=lambda: self.close_window(app), fg_color="#528868", text_color="#ffffff")

        # Placement des widgets
        label_titre.grid(row=0, column=0, columnspan=3, pady=(20, 10))
        label_consignes.grid(row=1, column=0, columnspan=3, pady=(10, 20))
        separator.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 20))

        text_sex.grid(row=3, column=0, padx=20, pady=(20, 2))
        self.combobox_sex.grid(row=4, column=0, padx=20, pady=(2, 20))

        text_bald.grid(row=5, column=0, padx=20, pady=(20, 2))
        self.combobox_bald.grid(row=6, column=0, padx=20, pady=(2, 20))

        text_cheveux.grid(row=7, column=0, padx=20, pady=(20, 2))
        self.combobox_cheveux.grid(row=8, column=0, padx=20, pady=(2, 20))

        text_texture.grid(row=9, column=0, padx=20, pady=(20, 2))
        self.combobox_texture.grid(row=10, column=0, padx=20, pady=(2, 20))

        text_corpu.grid(row=11, column=0, padx=20, pady=(20, 2))
        self.combobox_corpu.grid(row=12, column=0, padx=20, pady=(2, 20))

        text_lunettes.grid(row=13, column=0, padx=20, pady=(20, 2))
        self.combobox_lunettes.grid(row=14, column=0, padx=20, pady=(2, 20))

        text_age.grid(row=15, column=0, padx=20, pady=(20, 2))
        self.combobox_age.grid(row=16, column=0, padx=20, pady=(2, 20))

        text_pilosité.grid(row=17, column=0, padx=20, pady=(20, 2))
        self.combobox_pilosité.grid(row=18, column=0, padx=20, pady=(2, 20))

        button.grid(row=19, column=0, pady=(50, 20))

    def close_window(self, app):
        # Récupération des réponses
        reponses = {
            "Male": self.combobox_sex.get(),
            "Bald": self.combobox_bald.get(),
            "Brown_Hair": self.combobox_cheveux.get(),
            "Gray_Hair": self.combobox_cheveux.get(),
            "Black_Hair": self.combobox_cheveux.get(),
            "Blond_Hair": self.combobox_cheveux.get(),
            "Wavy_Hair": self.combobox_texture.get(),
            "Straight_Hair": self.combobox_texture.get(),
            "Chubby": self.combobox_corpu.get(),
            "Eyeglasses": self.combobox_lunettes.get(),
            "Young": self.combobox_age.get(),
            "Mustache": self.combobox_pilosité.get(),
            "Goatee": self.combobox_pilosité.get(),
            "No_Beard": self.combobox_pilosité.get()
        }
        if "Sélectionner" in reponses.values():
            messagebox.showerror("", "Veuillez répondre à toutes les questions.")
        else:
            converted_reponses = self.conversion_reponses(reponses)
            messagebox.showinfo("", "Vos réponses ont bien été enregistrées.")
            app.destroy()
        print(converted_reponses)
            
        return converted_reponses

    def conversion_reponses(self, reponses):
        conversions = {
        "Male": {"Homme": 1, "Femme": -1, "Je ne sais pas": 0},
        "Bald": {"Oui": 1, "Non": -1, "Je ne sais pas": 0},
        "Brown_Hair": {"Noirs": -1, "Bruns/Châtains": 1, "Blonds": -1, "Gris": -1, "Je ne sais pas": 0},
        "Gray_Hair": {"Noirs": -1, "Bruns/Châtains": -1, "Blonds": -1, "Gris": 1, "Je ne sais pas": 0},
        "Black_Hair": {"Noirs": 1, "Bruns/Châtains": -1, "Blonds": -1, "Gris": -1, "Je ne sais pas": 0},
        "Blond_Hair": {"Noirs": -1, "Bruns/Châtains": -1, "Blonds": 1, "Gris": -1, "Je ne sais pas": 0},
        "Bald": {"Noirs": -1, "Bruns/Châtains": -1, "Blonds": -1, "Gris": -1, "Je ne sais pas": 0},
        "Wavy_Hair": {"Lisses": -1, "Bouclés": 1, "Je ne sais pas": 0},
        "Straight_Hair": {"Lisses": 1, "Bouclés": -1, "Je ne sais pas": 0},
        "Chubby": {"Faible": -1, "Forte": 1, "Je ne sais pas": 0},
        "Eyeglasses": {"Oui": 1, "Non": -1, "Je ne sais pas": 0},
        "Young": {"Jeune": 1, "Agé": -1, "Je ne sais pas": 0},
        "Mustache": {"Oui": 0, "Non": -1, "Je ne sais pas": 0},
        "Goatee": {"Oui": 0, "Non": -1, "Je ne sais pas": 0},
        "No_Beard": {"Oui": -1, "Non": 1, "Je ne sais pas": 0},
        }
        
        # Conversion des réponses
        reponses_converted = {
            key: conversions[key].get(value, value)
            for key, value in reponses.items()
        }

        if reponses_converted.get("Bald") == 1:
            reponses_converted["Straight_Hair"] = -1
            reponses_converted["Wavy_Hair"] = -1

        return reponses_converted

    def displayGrid(self):
        self.grid.setHeight(self.photosFrame.winfo_height())
        self.grid.setWidth(self.photosFrame.winfo_width())

        self.grid.displayImages()
    
    def nextGen(self):
        print("Pas implémenté")
    
    def previousGen(self):
        print("Pas implémenté")
    
    def displayExportWindow(self):
        exp_window = CTkToplevel(self.root)
        exp_window.title("Exporter")
        exp_window.geometry("450x300")
        exp_window.grab_set()

    def displayParameterWindow(self):
        # Fenêtre qui correspond aux différents paramètres de l'applications.
        change = True
        self.disp_window = CTkToplevel(self.root)
        self.disp_window.title("Paramètres")
        self.disp_window.geometry("300x150")
        self.disp_window.grab_set()

        topFrame = CTkFrame(self.disp_window)
        midFrame = CTkFrame(self.disp_window)
        bottomFrame = CTkFrame(self.disp_window)

        title = CTkLabel(topFrame, text="Paramètres", font=("Arial", 30), wraplength=480)

        textMP = CTkLabel(midFrame,text="Sélection multiple : ")
        self.checkVarMP = StringVar(value="on")
        checkboxSelecMP = CTkCheckBox(midFrame, text='',command=lambda : self.grid.setUniqueSelection(self.checkVarMP),
                                     variable=self.checkVarMP, onvalue=True, offvalue=False)
        
        textImage = CTkLabel(midFrame, text="Nombre d'images par générations : ")
        self.nbGenImages = IntVar(value=6)
        spinboxImages = CTkSpinbox(midFrame, variable = self.nbGenImages, min_value = 4, max_value= 9, width=60, height=15,border_width=0)

        textFus = CTkLabel(midFrame,text="Méthode de fusion : ")
        self.checkVarFus = StringVar(value="BLEND")
        comboFus = CTkCheckBox(midFrame, text='', values = ['BLEND', 'VAE'], command=lambda : self.grid.setFusionMethod(self.checkVarFus),
                                     variable=self.checkVarFus)

        validateButton = CTkButton(bottomFrame, text="Sauvegarder", command = lambda window=self.disp_window: print(self.disp_window.winfo_geometry()))

        self.disp_window.grid_columnconfigure((0,1,2), weight=1) 
        topFrame.grid_columnconfigure(0, weight=1)
        midFrame.grid_columnconfigure((0,1,2), weight=1)
        bottomFrame.grid_columnconfigure(0, weight=1)
        topFrame.grid(row = 0, sticky="nsew")
        midFrame.grid(row = 1, sticky="nsew")
        bottomFrame.grid(row = 2, sticky="nsew")

        title.grid(column = 0, row = 0, sticky="nsew")
        textMP.grid(column = 0, row = 0, sticky="nsew")
        checkboxSelecMP.grid(column = 1, row = 0, sticky="nsew")
        textImage.grid(column = 0, row=1, sticky="nsew")
        spinboxImages.grid(column = 1, row = 1, sticky="nsew")
        validateButton.grid(column= 0, padx = 20, pady=10)
        self.disp_window.protocol("WM_DELETE_WINDOW", lambda val=change : self.destroyParam(change))
    
    def destroyParam(self, changed):
        if changed:
            messagebox.askquestion("" , "Souhaitez-vous continuer sans enregistrer ?")
        self.disp_window.destroy()
        



if __name__ == "__main__":
    test = IHM()