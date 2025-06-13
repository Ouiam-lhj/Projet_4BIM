# Zencoder's Profiler

## Comment utiliser l'application 

**Requis**

- Afin de pouvoir utiliser l'interface, il est nécessaire d'avoir un appareil avec un système d'exploitation Linux ou MacOS (le fichier de package requis est différents pour les utilisateurs MacOS, voir ci-dessous).
- Nous vous recommandons l'installation d'une machine virtuel
- Nous vous recommandons une version une version python entre 3.9 et 3.11

Un problème a été identifié sur Windows avec cmake, causant des erreurs lors de l'installation de dlib.

### Installation et création de l'environnement virtuel

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

Il faudra ensuite l'activer afin d'isoler l'installation des packages :

```bash
$source virtual_environment/bin/activate
```

L'installation des package se fait à l'aide du fichier `requirements.txt` pour les utilisateurs de Linux et `requirements_mos.txt` pour les utilisateurs de MacOS :

```bash
# Linux
$pip install -r requirements.txt
```

```bash
# MacOS
$pip install -r requirements_mos.txt
```

Il est ensuite de lancer directement l'interface à l'aide de la commande suivante (python ou python3 selon la version de python que vous possédez) :

```bash
$python3 src/Zencoder_profiler/IHM.py
```

## Liens database

La base de donnée d'images CelebA a été utilisée pour alimenter la base de donnée fourni avec le logiciel. Pour obtenir plus d'images, veuillez consulter le site de Celeb A :


Lien du dataset CelebA : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html


lien drive data: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg


**Attention** : Les images d'individus de biais sont à éviter puisqu'elles causent des problèmes.