from customtkinter import *
import tkinter as tk
import customtkinter
from tkinter import messagebox
from CTkListbox import *
from PIL import Image, ImageTk
import numpy as np
from CTkSpinbox import * 

#pip install customtkinter
#pip install CTkListbox
#pip install tkinter
#pip install PIL
#pip install CTkSpinbox

class ImageDisplayError(Exception):
    pass

class DynamicGrid():
    
    def __init__(self, links, parentFrame, grid_width, grid_height, margin=0.01):
        self.parentFrame = parentFrame
        self.links = links
        self.loadImages(links)
        self.width = grid_width
        self.height = grid_height
        self.margin = margin
        self.frames = []
        self.selected_images = []
        self.rows = 0
        self.columns = 0
        self.isEmpty = True
    
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

    def ToCTkImage(self, s):
        self.images = list(map(lambda x : CTkImage(light_image=x, dark_image=x, size = (s,s)),self.images))
    
    def getImages(self):
        return self.images
    
    def getImage(self, k):
        if k > len(self.images):
            raise IndexError
        return self.images[k]

    def displayImages(self):
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
        self.loadImages(self.links)
        self.ToCTkImage(images_size)

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
        self.root.mainloop()

    def homePage(self):
        self.images = ["000001.jpg","000002.jpg","000003.jpg","000004.jpg","000005.jpg","000006.jpg"]

        self.root.title("Le profiler des zencoders")

        self.principalMainframe = CTkFrame(self.root, fg_color="#38393b", border_width = 0)
        self.menuMainframe = CTkFrame(self.root, width= 200)
        self.titleFrame = CTkFrame(self.principalMainframe, fg_color="#00FF00", height = 70)
        self.photosFrame = CTkFrame(self.principalMainframe, fg_color="#0000FF", height = 400)

        self.buttonsFrame=CTkFrame(self.principalMainframe, fg_color="#FF0000", height = 50)
        self.leftSideButtonFrame = CTkFrame(self.buttonsFrame, fg_color="#FFC0CB", height = 50)
        self.middleSideButtonFrame = CTkFrame(self.buttonsFrame, fg_color="#FFFF00", height = 50)
        self.rightSideButtonFrame = CTkFrame(self.buttonsFrame, fg_color="#FF4500", height = 50)


        self.menuFormButton = CTkButton(self.menuMainframe, text="Formulaire", command=self.displayFormulaire, fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'])
        self.menuExportButton = CTkButton(self.menuMainframe, text="Exporter", command=self.displayExportWindow, fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'])
        self.menuParamButton = CTkButton(self.menuMainframe, text="Paramètres", command=self.displayParameterWindow, fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'])
        self.photo = CTkButton(self.menuMainframe, text="Test", command= lambda : self.displayGrid(), fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'])
        self.grid = DynamicGrid(self.images, self.photosFrame, self.photosFrame.winfo_width() ,self.photosFrame.winfo_height())
        self.newGenButton = CTkButton(self.middleSideButtonFrame, text = "Nouvelle génération", command = lambda : self.grid.get_selected_images(), fg_color="transparent", hover_color=['#e4e4eb', '#3a3b3d'])
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

        self.menuFormButton.pack(fill="x", pady=10)
        self.menuExportButton.pack(fill="x", pady=10)
        self.menuParamButton.pack(fill="x", pady=10)
        self.photo.pack(fill="x", pady=15)


    def displayFormulaire(self):
        app = CTkToplevel(self.root)
        app.title("Questionnaire")
        app.geometry("520x600")
        app.configure(bg="#f5f5f5")
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
        text_sex = customtkinter.CTkLabel(scroll, text="Quel était le sexe de l'individu ?", font=("Arial", font_size_questions), text_color=text_color)
        text_peau = customtkinter.CTkLabel(scroll, text="Quelle était la couleur de peau de l'individu ?", font=("Arial", font_size_questions), text_color=text_color)
        text_cheveux = customtkinter.CTkLabel(scroll, text="Quelle était la couleur des cheveux de l'individu ?", font=("Arial", font_size_questions), text_color=text_color)
        text_texture = customtkinter.CTkLabel(scroll, text="Quel était la texture de cheveux de l'individu ?", font=("Arial", font_size_questions), text_color=text_color)
        text_corpu = customtkinter.CTkLabel(scroll, text="Quelle était la corpulence de l'individu ?", font=("Arial", font_size_questions), text_color=text_color)
        text_lunettes = customtkinter.CTkLabel(scroll, text="L'individu portait-il des lunettes ?", font=("Arial", font_size_questions), text_color=text_color)
        text_age = customtkinter.CTkLabel(scroll, text="L'individu vous paraissait plutôt :", font=("Arial", font_size_questions), text_color=text_color)
        text_pilosité = customtkinter.CTkLabel(scroll, text="Quelle était la pilosité de l'individu", font=("Arial", font_size_questions), text_color=text_color)


        # Réponses (Création et set)
        self.combobox_sex = customtkinter.CTkComboBox(scroll, values=["Homme", "Femme", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_sex.set("Sélectionner")

        self.combobox_peau = customtkinter.CTkComboBox(scroll, values=["Pâle", "Foncée", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_peau.set("Sélectionner")

        self.combobox_cheveux = customtkinter.CTkComboBox(scroll, values=["Noirs/Bruns/Châtains", "Blonds", "Roux", "Chauve", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_cheveux.set("Sélectionner")

        self.combobox_texture = customtkinter.CTkComboBox(scroll, values=["Lisses", "Bouclés", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_texture.set("Sélectionner")

        self.combobox_corpu = customtkinter.CTkComboBox(scroll, values=["Faible", "Forte", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_corpu.set("Sélectionner")

        self.combobox_lunettes = customtkinter.CTkComboBox(scroll, values=["Oui", "Non", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_lunettes.set("Sélectionner")

        self.combobox_age = customtkinter.CTkComboBox(scroll, values=["Jeune", "Agé", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_age.set("Sélectionner")

        self.combobox_pilosité = customtkinter.CTkComboBox(scroll, values=["Absente", "Bouc", "Moustache", "Bouc et moustache", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_pilosité.set("Sélectionner")

        # Bouton de validation
        button = customtkinter.CTkButton(scroll, text="Valider", font=("Arial", font_size_questions), width=200, height=30, command=lambda: self.close_window(app), fg_color="#528868", text_color="#ffffff")

        # Placement des widgets
        label_titre.grid(row=0, column=0, columnspan=3, pady=(20, 10))
        label_consignes.grid(row=1, column=0, columnspan=3, pady=(10, 20))
        separator.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 20))

        text_sex.grid(row=3, column=0, padx=20, pady=(20, 2))
        self.combobox_sex.grid(row=4, column=0, padx=20, pady=(2, 20))

        text_peau.grid(row=5, column=0, padx=20, pady=(20, 2))
        self.combobox_peau.grid(row=6, column=0, padx=20, pady=(2, 20))

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
            "Sexe": self.combobox_sex.get(),
            "Couleur de peau": self.combobox_peau.get(),
            "Couleur des cheveux": self.combobox_cheveux.get(),
            "Texture des cheveux": self.combobox_texture.get(),
            "Corpulence": self.combobox_corpu.get(),
            "Lunettes": self.combobox_lunettes.get(),
            "Âge": self.combobox_age.get(),
            "Pilosité": self.combobox_pilosité.get()
        }
        if "Sélectionner" in reponses.values():
            messagebox.showerror("", "Veuillez répondre à toutes les questions.")
        else:
            messagebox.showinfo("", "Vos réponses ont bien été enregistrées.")
            app.destroy()
            
        return reponses

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
        self.checkVarMP = customtkinter.StringVar(value="on")
        checkboxSelecMP = customtkinter.CTkCheckBox(midFrame, text='',command=lambda : print(self.checkVarMP.get()),
                                     variable=self.checkVarMP, onvalue="on", offvalue="off")
        
        textImage = CTkLabel(midFrame, text="Nombre d'images par générations : ")
        self.nbGenImages = customtkinter.IntVar(value=6)
        spinboxImages = CTkSpinbox(midFrame, variable = self.nbGenImages, min_value = 4, max_value= 9, width=60, height=15,border_width=0)

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