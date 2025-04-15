# Projet_4BIM

# Comment utiliser l'application 

**Requis**

- Afin de pouvoir utiliser l'interface, il est nécessaire d'avoir un appareil avec un système d'exploitation Linux.
- Nous vous recommandons l'installation d'une machine virtuel
- Nous vous recommandons une version une version python entre 3.9 et 3.11

## Installation et création de l'environnement virtuel

Dans un premier temps, cloner le git à l'aide de la fonction suivante
```bash
$git clone https://github.com/Ouiam-lhj/Projet_4BIM.git <name>
```

Il faut ensuite se placer dans le git cloner afin de pouvoir l'utiliser.
```bash
$cd <name>
```

Il faut alors créer un environnement virtuel :
```bash
$python3 -m venv virtual_environment
```

## Package à installer si création d'un environnement manuellement
Package à installer sur Windows :
```
pip install customtkinter
pip install CTkListbox
pip install Pillow
pip install CTkSpinbox
pip install numpy
pip install matplotlib
pip install opencv-python
pip install dlib
pip install pandas
pip install tensorflow
pip install scikit-learn
```

Package à installer sur MacOS
```
pip install customtkinter
pip install CTkListbox
pip install Pillow
pip install CTkSpinbox
pip install numpy
pip install matplotlib
pip install pandas
pip install opencv-python
pip install dlib
pip install tensorflow-macos
pip install tensorflow-metal
pip install scikit-learn
```

Un environnement virtuel a été crée localement avec les packages, il vous suffit de l'activer :

```bash
$source virtual_environment/bin/activate
```

Il est ensuite de lancer directement l'interface à l'aide de la commande suivante (python ou python3 selon la version de python que vous possédez) :

```bash
$python3 src/Zencoder_profiler/IHM.py
```

# Liens database

Pour augmenter les photos disponibles sur l'interfaces, vous pouvez directement les télécharger depuis le site de celebA

Lien du dataset CelebA : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
lien drive data: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg
