from customtkinter import *
import tkinter as tk
import customtkinter
from tkinter import messagebox
from CTkListbox import *
from PIL import Image, ImageTk
import numpy as np

#pip install customtkinter
#pip install CTkListbox
#pip install tkinter
#pip install PIL

class DynamicGrid():
    
    def __init__(self, links, parentFrame, grid_width, grid_height, margin=0.01):
        self.parentFrame = parentFrame
        self.links = links
        self.loadImages(links)
        self.width = grid_width
        self.height = grid_height
        self.margin = margin
        self.frames = []
        self.rows = 0
        self.columns = 0
    

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
        images_displayed=0

        print(self.images)
        nb_images = len(self.images)
        print(nb_images)
        self.get_grid_dimensions(nb_images)

        print("Columns = " + str(self.columns))
        print("Rows = " + str(self.rows))
        width_frame = self.width/self.rows
        height_frame = self.height/self.columns
        
        images_size = max(width_frame/self.columns * (1 - self.margin), height_frame/self.rows * (1 - self.margin))
        self.ToCTkImage(images_size)

        for i in range(self.columns):
            print('i = ', i)
            currentFrame = CTkFrame(self.parentFrame, width=width_frame, height=height_frame, fg_color="#02D8E8")
            currentFrame.pack(expand=True, fill="both", side="left")
            self.frames.append(currentFrame)
            for j in range(images_displayed, images_displayed + self.rows):
                print('j = ', j)
                button = CTkLabel(currentFrame, image= self.images[j],width=images_size, height=images_size)
                button.pack(side = TOP)
                button.image = self.images[j]
            images_displayed += self.rows

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

        self.menuFormButton = CTkButton(self.menuMainframe, text="Formulaire", command=self.displayFormulaire, fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'])
        self.photo = CTkButton(self.menuMainframe, text="Pour toi Mathis", command= lambda : self.displayGrid(), fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'])
        
        self.principalMainframe.pack(expand=True, fill="both", side="right")
        self.menuMainframe.pack(fill="y", side="left")
        self.titleFrame.pack(expand=True, fill="both", side="top")
        self.photosFrame.pack(expand=True, fill="both", side="top")
        self.buttonsFrame.pack(expand=True, fill="both", side="top")

        self.menuFormButton.pack(fill="x", pady=10)
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
        self.grid = DynamicGrid(self.images, self.photosFrame, self.photosFrame.winfo_width() ,self.photosFrame.winfo_height())
        self.grid.displayImages()


if __name__ == "__main__":
    test = IHM()