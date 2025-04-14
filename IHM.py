from customtkinter import *
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from CTkSpinbox import * 
import io
from code_gen_blend import *
from autoencoder import *

#pip install customtkinter
#pip install CTkListbox
#pip install tkinter
#pip install Pillow
#pip install CTkSpinbox

class ImageDisplayError(Exception):
    pass

class SelectionError(Exception):
    pass

class MethodError(Exception):
    pass
class DynamicGrid():
    
    
    def __init__(self, links, parentFrame, grid_width, grid_height, margin=0.01, multi_selection = True):
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
        self.multi_selection = multi_selection
        self.fusionMethod = "BLEND"
        self.nbImage = 7
        self.images_history = []
        self.index_image_history = -1
    
    def figsToImage(self):
        for fig in self.figures :
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            img = Image.open(buf)
    
    def getIndexHistory(self):
        return self.index_image_history
    
    def maxIndexHistory(self):
        return len(self.images_history)
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

    def resizeImages(self, images):
        images = list(map(lambda x : x.resize((250, 250), Image.LANCZOS), images))
        return images
    
    def loadImages(self, links):
        self.images = list(map(lambda x : Image.open(x), links))

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
            arrays = apply_random_blending(pil_images,self.nbImage)
            self.figures = list(map(lambda x: Image.fromarray(x), arrays))
            self.figures = self.resizeImages(self.figures)
            print(self.figures[0].size)
            print("FIGURES : {}".format(self.figures))
        # On a obtenue les (ou la dans le cadre du premier test)
        elif (self.fusionMethod == "VAE"):
            print("Il me faut les mutations pour réaliser la fusions des codes")
            pil_images = vae_generate_mutated_images(var_encoder, var_decoder, self.selected_images, new_to_show=self.nbImage, mutation_strength=0.5)
            self.figures = self.resizeImages(self.figures)
        else:
            raise MethodError("Un mot clef incorrect a été utilisé pour le passage de génération")
    
        self.destroyGrid()
        self.displayImages(source="FIGURES", add_to_history=True)
        self.selected_images = []


        # Le résultat est stocké dans une variable et on display les images.
    def displayImages(self, source = "LINK", add_to_history = False):
        # La frame doit être un objet CTkFrame
        # images est une liste d'images
        if (self.images == []):
            raise ImageDisplayError
        
        if (self.isEmpty == False):
            self.destroyGrid()

        images_displayed=0

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
        elif source == "IMAGE":
            pass
        else:
            return

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
            print("Ajout à l'historique")
            self.images_history.append([self.images.copy(), self.selected_images.copy()])
            self.index_image_history += 1
    
    def previousImages(self):
        if self.index_image_history <= 0:
            raise ValueError("Impossible de charger des images avant la première génération")
        self.images = self.images_history[self.index_image_history - 1][0]
        self.displayImages(source="IMAGE", add_to_history=False)
        self.index_image_history -= 1
    
    def nextImages(self):
        if self.index_image_history == (len(self.images_history) - 1):
            raise ValueError("Impossible de charger des images qui n'ont pas encore été générée")
        self.images = self.images_history[self.index_image_history + 1][0]
        self.displayImages(source="IMAGE", add_to_history=False)
        self.index_image_history += 1

    def resetImages(self):
        self.selected_images = self.images_history[self.index_image_history - 1][1]
        self.algoGen()
        self.index_image_history += 1
        print(self.index_image_history)
        print(len(self.images_history))

    def destroyGrid(self):
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
        self.images = ["image_0.png","image_2.png","image_4.png","image_6.png","image_8.png","image_10.png"]

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
        self.photo = CTkButton(self.menuMainframe, text="Test", command= lambda : self.displayGrid(), fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'], text_color='#333333')
        self.grid = DynamicGrid(self.images, self.photosFrame, self.photosFrame.winfo_width() ,self.photosFrame.winfo_height())
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
        self.photo.pack(fill="x", pady=15)

    def displayFormulaire(self):
        app = CTkToplevel(self.root)
        app.title("Questionnaire")
        app.geometry("520x600")
        app.configure(bg="#f5f5f5")
        app.grid_rowconfigure(0, weight=1)
        app.grid_columnconfigure(0, weight=1)

        # Scrollable frame
        scroll = CTkScrollableFrame(app, width=500, height=600, fg_color="#ffffff")
        scroll.grid(row=0, column=0, columnspan=3, sticky="nsew")
        scroll.grid_rowconfigure(0, weight=1)
        scroll.grid_columnconfigure(0, weight=1)
        app.grab_set()
        
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
        text_cheveux = CTkLabel(scroll, text="Quelle était la couleur des cheveux de l'individu ?", font=("Arial", font_size_questions), text_color=text_color)
        text_texture = CTkLabel(scroll, text="Quel était la texture de cheveux de l'individu ?", font=("Arial", font_size_questions), text_color=text_color)
        text_age = CTkLabel(scroll, text="L'individu vous paraissait plutôt :", font=("Arial", font_size_questions), text_color=text_color)


        # Réponses (Création et set)
        self.combobox_sex = CTkComboBox(scroll, values=["Homme", "Femme", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_sex.set("Sélectionner")

        self.combobox_cheveux = CTkComboBox(scroll, values=["Noirs", "Bruns/Châtains", "Blonds", "Gris", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_cheveux.set("Sélectionner")

        self.combobox_texture = CTkComboBox(scroll, values=["Lisses", "Bouclés", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_texture.set("Sélectionner")

        self.combobox_age = CTkComboBox(scroll, values=["Jeune", "Agé", "Je ne sais pas"], width=200, height=20, font=("Arial", font_size_questions), state="readonly", fg_color="#ffffff", text_color=text_color)
        self.combobox_age.set("Sélectionner")

        # Bouton de validation
        button = CTkButton(scroll, text="Valider", font=("Arial", font_size_questions), width=200, height=30, command=lambda: self.close_window(app), fg_color="#528868", text_color="#ffffff")

        # Placement des widgets
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
        # Récupération des réponses
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
            app.destroy()

        sample = self.chose_first_sample_photo(self.get_photos_matching_form(converted_reponses))
            
        return sample

    def conversion_reponses(self, reponses):
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
        
        # Conversion des réponses en valeurs numériques
        for key in reponses:
            if key in conversions and reponses[key] in conversions[key]:
                reponses[key] = conversions[key][reponses[key]]
            else:
                reponses[key] = 0 

        return reponses

    def get_photos_matching_form(self, reponses):
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
        sample = df_form.sample(n=6)
        return sample.index


    def displayGrid(self):
        if self.consignes_label.winfo_exists():
            self.consignes_label.pack_forget()
        print(self.root.winfo_geometry())
        self.grid.setHeight(self.photosFrame.winfo_height())
        self.grid.setWidth(self.photosFrame.winfo_width())


        self.grid.displayImages(add_to_history=True)
    
    def nextGen(self):
        self.grid.nextImages()
        self.updateButtonStatus()
    
    def previousGen(self):
        self.grid.previousImages()
        self.updateButtonStatus()

    def updateButtonStatus(self):
        if (self.grid.maxIndexHistory() == 0):
            self.newGenButton.configure(state = "disable")
            self.previousGenButton.configure(state = "disable")
        else:
            if (self.grid.getIndexHistory() == 0):
                self.newGenButton.configure(state = "enable")
                self.previousGenButton.configure(state="disable")
            elif (self.grid.getIndexHistory() == self.grid.maxIndexHistory()):
                self.newGenButton.configure(state = "disable")
                self.previousGenButton.configure(state="enable")

    def displayParameterWindow(self):
        self.disp_window = CTkToplevel(self.root)
        self.disp_window.title("Paramètres")
        self.disp_window.geometry("360x240")
        self.disp_window.resizable(False, False)

        # Frames
        topFrame = CTkFrame(self.disp_window)
        midFrame = CTkFrame(self.disp_window)
        bottomFrame = CTkFrame(self.disp_window)

        topFrame.pack(pady=10, fill="x")
        midFrame.pack(padx=20, pady=10, fill="both", expand=True)
        bottomFrame.pack(pady=10)

        title = CTkLabel(topFrame, text="Paramètres", font=("Arial", 24, "bold"))
        title.pack()

        # Variables temporaires
        self.temp_checkVarMP = BooleanVar(value=self.grid.multi_selection)
        print(self.temp_checkVarMP.get())
        self.temp_nbGenImages = IntVar(value=self.grid.nbImage)
        self.temp_checkVarFus = StringVar(value=self.grid.fusionMethod)

        # Sélection multiple
        textMP = CTkLabel(midFrame, text="Sélection multiple :")
        checkboxSelecMP = CTkCheckBox(midFrame, text='', variable=self.temp_checkVarMP)

        # Nombre d’images
        textImage = CTkLabel(midFrame, text="Images par génération :")
        spinboxImages = CTkSpinbox(midFrame, variable=self.temp_nbGenImages, min_value=4, max_value=9,
                                width=80, height=28, border_width=1)
        spinboxImages.set(self.grid.nbImage)

        # Méthode de fusion
        textFus = CTkLabel(midFrame, text="Méthode de fusion :")
        comboFus = CTkComboBox(midFrame, values=['BLEND', 'VAE'], variable=self.temp_checkVarFus)

        # Placement
        textMP.grid(row=0, column=0, sticky="w", pady=5)
        checkboxSelecMP.grid(row=0, column=1, sticky="e", pady=5)
        textImage.grid(row=1, column=0, sticky="w", pady=5)
        spinboxImages.grid(row=1, column=1, sticky="e", pady=5)
        textFus.grid(row=2, column=0, sticky="w", pady=5)
        comboFus.grid(row=2, column=1, sticky="e", pady=5)

        # Boutons
        validateButton = CTkButton(bottomFrame, text="Sauvegarder", command=self.saveParameters)
        cancelButton = CTkButton(bottomFrame, text="Annuler", fg_color="grey", hover_color="#999",
                                command=self.confirmCloseWithoutSave)

        validateButton.grid(row=0, column=0, padx=10)
        cancelButton.grid(row=0, column=1, padx=10)

        # Gestion fermeture
        self.disp_window.protocol("WM_DELETE_WINDOW", self.confirmCloseWithoutSave)
        self.disp_window.grab_set()
        self.disp_window.focus_force()


    def confirmCloseWithoutSave(self):
        response = messagebox.askyesno("Confirmation", "Souhaitez-vous quitter sans sauvegarder ?")
        if response:
            self.disp_window.grab_release()
            self.disp_window.destroy()
        else:
            self.disp_window.focus_force()


    def saveParameters(self):
        print(self.temp_checkVarMP.get())
        self.grid.multi_selection = self.temp_checkVarMP.get()
        self.grid.nbImage = self.temp_nbGenImages.get()
        self.grid.fusionMethod = self.temp_checkVarFus.get()
        self.disp_window.grab_release()
        self.disp_window.destroy()
    
    def displayExportWindow(self):

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
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path_var.set(folder)


    def export_images(self):
        folder = self.folder_path_var.get()
        if not folder:
            messagebox.showerror("Erreur", "Sélectionnez un dossier pour l'export !")
            return
        
        export_format = self.format_var.get()

        if not self.grid.selected_images :
            messagebox.showerror("Erreur", "Pas d'images sélectionnée pour l'export !")
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