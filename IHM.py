from customtkinter import *
import tkinter as tk
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

        self.menuFormButton = CTkButton(self.menuMainframe, text="Formulaire", command= lambda : self.displayGrid(), fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'])
        
        self.principalMainframe.pack(expand=True, fill="both", side="right")
        self.menuMainframe.pack(fill="y", side="left")
        self.titleFrame.pack(expand=True, fill="both", side="top")
        self.photosFrame.pack(expand=True, fill="both", side="top")
        self.buttonsFrame.pack(expand=True, fill="both", side="top")

        self.menuFormButton.pack(fill="x", pady=10)

    def displayFormulaire(self):
        print("Code du formulaire ici")
        return

    def displayGrid(self):
        self.grid = DynamicGrid(self.images, self.photosFrame, self.photosFrame.winfo_width() ,self.photosFrame.winfo_height())
        self.grid.displayImages()


if __name__ == "__main__":
    test = IHM()