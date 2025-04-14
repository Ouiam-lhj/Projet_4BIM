import os
import pandas as pd

# Charger le fichier CSV (assurez-vous qu'il est bien en UTF-8 ou utilisez un autre encodage si nécessaire)
csv_file = 'final_dataset.csv'  # Remplace ce chemin par celui de ton fichier CSV
df = pd.read_csv(csv_file, encoding='latin1')  # Essaie 'latin1' si UTF-8 pose problème


indices = df.iloc[:, 0].astype(str).tolist()
print(len(indices))

image_folder = 'img_align_celeba'  # Remplace ce chemin par celui de ton dossier d'images

# # Lister les fichiers dans le dossier d'images
images = os.listdir(image_folder)
print(len(images))


# import shutil

# # Dossier où déplacer les images non souhaitées
# removed_folder = 'img_align_celeba_removed'

# # Créer le dossier pour les images supprimées si ce n'est pas déjà fait
# if not os.path.exists(removed_folder):
#     os.makedirs(removed_folder)


# # Créer un set des indices pour une recherche plus rapide
# indices_set = set(indices)

# # Parcourir toutes les images
# for image in images:
#     if image not in indices_set:
#         # Si l'image n'est pas dans indices, la déplacer
#         src = os.path.join(image_folder, image)
#         dst = os.path.join(removed_folder, image)
#         shutil.move(src, dst)

# print(f"Les images qui ne sont pas dans 'indices' ont été déplacées vers '{removed_folder}'")
