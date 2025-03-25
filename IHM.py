from customtkinter import *
import tkinter as tk
from CTkListbox import *

#pip install customtkinter
#pip install CTkListbox
#pip install tkinter

class IHM():

    def __init__(self):
        self.root = CTk()
        self.homePage()
        self.root.mainloop()

    def homePage(self):

        self.root.title("Le profiler des zencoders")

        self.principalMainframe = CTkFrame(self.root, fg_color="#38393b", border_width = 0)
        self.menuMainframe = CTkFrame(self.root, width= 200)
        self.titleFrame = CTkFrame(self.principalMainframe, fg_color="#00FF00", height = 70)
        self.photosFrame = CTkFrame(self.principalMainframe, fg_color="#0000FF", height = 400)
        self.buttonsFrame=CTkFrame(self.principalMainframe, fg_color="#FF0000", height = 50)

        self.menuFormButton = CTkButton(self.menuMainframe, text="Formulaire", command= lambda : print(self.root.winfo_geometry()), fg_color="transparent", border_width = 0, hover_color=['#e4e4eb', '#3a3b3d'])
        
        self.principalMainframe.pack(expand=True, fill="both", side="right")
        self.menuMainframe.pack(fill="y", side="left")
        self.titleFrame.pack(expand=True, fill="both", side="top")
        self.photosFrame.pack(expand=True, fill="both", side="top")
        self.buttonsFrame.pack(expand=True, fill="both", side="top")

        self.menuFormButton.pack(fill="x", pady=10)
    
    def displayFormulaire(self):
        print("Code du formulaire ici")
        return

if __name__ == "__main__":
    test = IHM()
