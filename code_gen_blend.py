#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 13:08:43 2025

@author: ouiamelhajji
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import pandas as pd
import cv2
#!pip install dlib
import dlib
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import math

list_attr=pd.read_csv("list_attr_celeba.txt" , sep=r"\s+", header = 0)
pos_attr=pd.read_csv("list_landmarks_align_celeba.txt" , sep=r"\s+", header = 0)


#image_1 = Image.open("img_align_celeba/000001.jpg")
#image_2 = Image.open("img_align_celeba/000002.jpg")
#image_3 = Image.open("img_align_celeba/000003.jpg")
#image_4 = Image.open("img_align_celeba/000004.jpg")
#image_5 = Image.open("img_align_celeba/000005.jpg")

#PR RUN SUR ORDI OUIAM
#PR RUN SUR ORDI OUIAM
image_1 = Image.open("/Users/ouiamelhajji/Documents/INSA/4A/S2/devweb/000001.jpg")
image_2 = Image.open("/Users/ouiamelhajji/Documents/INSA/4A/S2/devweb/000002.jpg")
image_3 = Image.open("/Users/ouiamelhajji/Documents/INSA/4A/S2/devweb/000003.jpg")
image_4 = Image.open("/Users/ouiamelhajji/Documents/INSA/4A/S2/devweb/000004.jpg")
image_5 = Image.open("/Users/ouiamelhajji/Documents/INSA/4A/S2/devweb/000005.jpg")
image_10 = Image.open("/Users/ouiamelhajji/Documents/INSA/4A/S2/devweb/000010.jpg")



list_image = [image_1, image_2, image_3, image_4, image_5]

points_1 = np.array(pos_attr.iloc[0].to_frame().T).reshape(5, 2)
points_2 = np.array(pos_attr.iloc[1].to_frame().T).reshape(5, 2)
points_3 = np.array(pos_attr.iloc[2].to_frame().T).reshape(5, 2)
points_4 = np.array(pos_attr.iloc[3].to_frame().T).reshape(5, 2)
# Séparer les coordonnées X et Y
x1, y1 = points_1[:, 0], points_1[:, 1]
x2, y2 = points_2[:, 0], points_2[:, 1]
x3, y3 = points_3[:, 0], points_3[:, 1]
x4, y4 = points_4[:, 0], points_4[:, 1]


#plt.imshow(image_1)
#plt.scatter(x1, y2, c='red', marker='o', s=6)  # Points en rouge
#plt.imshow(image_2)
#plt.scatter(x2, y2, c='red', marker='o', s=6)  # Points en rouge
#plt.imshow(image_3)
#plt.scatter(x3, y3, c='red', marker='o', s=6)  # Points en rouge
#plt.imshow(image_4)
#plt.scatter(x4, y4, c='red', marker='o', s=6)  # Points en rouge
#plt.axis("on")  # Garde les axes activés (ou plt.axis("off") pour masquer)

# Afficher le tout

def align_face(img, landmarks_init, landmarks_fin, size=(256, 256)):
    """Aligne un visage en utilisant une transformation homographique basée sur les 68 landmarks."""
    if isinstance(img, Image.Image):
        img = np.array(img)
    init_points = np.float32(landmarks_init)
    fin_points = np.float32(landmarks_fin)
    matrix, _ = cv2.findHomography(init_points, fin_points)
    aligned_face = cv2.warpPerspective(img, matrix, size)
    return aligned_face

def condition_retourner(image1, image2):
    """Flip une des deux images si les visages sont tournés dans une direction différente."""
    position_landmarks_1 = [get_landmarks(image1)[position] for position in [30, 39, 42]]
    position_landmarks_2 = [get_landmarks(image2)[position] for position in [30, 39, 42]]

    centre1 = int((position_landmarks_1[1][0] + position_landmarks_1[2][0])/2)
    centre2 = int((position_landmarks_2[1][0] + position_landmarks_2[2][0])/2)

    delta = 5

    if ((centre1-position_landmarks_1[0][0]) < -delta and (centre2-position_landmarks_2[0][0]) > delta) or ((centre1-position_landmarks_1[0][0]) > delta and (centre2-position_landmarks_2[0][0]) < -delta):
        image2_retourne = retourner_visage(image2)
        return image1, image2_retourne
    return image1, image2


def retourner_visage(image):
    """Retourne une image selon un axe vertical."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image_retourne = ImageOps.mirror(image)
    return image_retourne

def get_landmarks(image):
    """Récupère les 68 points caractéristiques du visage grâce au modèle de dlib."""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    return np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])

def blend_faces(face1, face2, alpha=0.5):
    """Fusionne deux visages avec un mélange pondéré (taux alpha)."""
    landmarks1 = get_landmarks(face1)
    landmarks2 = get_landmarks(face2)

    if landmarks1 is None or landmarks2 is None:
        print("Erreur : Impossible de détecter les landmarks pour au moins un visage.")
        return None, None
    avg_pos_attr = np.mean([landmarks1, landmarks2], axis=0).astype(int)

    aligned_face_1 = align_face(face1, landmarks1, avg_pos_attr)
    aligned_face_2 = align_face(face2, landmarks2, avg_pos_attr)

    blended_face = cv2.addWeighted(aligned_face_1, alpha, aligned_face_2, 1 - alpha, 0)
    blended_face_pil = Image.fromarray(blended_face)

    return blended_face_pil, avg_pos_attr, blended_face
#print(blend_faces(image_1, image_2, alpha=0.45))


def draw_landmarks(image):
    """Dessine les 68 points caractéristiques sur l'image."""
    image = np.array(image)
    landmarks = get_landmarks(image)
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    return image
#print(get_landmarks(image_1))


def apply_blur_around_face(image, landmarks, blur_strength):
    """Applique un flou autour du visage en utilisant les landmarks."""
    image = np.array(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    center = np.mean(landmarks, axis=0).astype(int)
    radius = int(np.max(np.linalg.norm(landmarks - center, axis=1))) + 20
    cv2.circle(mask, tuple(center), radius, (255), thickness=-1)

    blurred_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    masked_face = cv2.bitwise_and(image, image, mask=mask)
    blurred_background = cv2.bitwise_and(blurred_image, blurred_image, mask=255-mask)
    combined_image = cv2.add(masked_face, blurred_background)

    return combined_image

def crop(face):
    """Enlève les bordures noires."""
    if isinstance(face, np.ndarray):
        face = Image.fromarray(face)
    face = face.crop((0, 0, 178, 218))

    return face
def mix_main(face1, face2, alpha):
    blended_face = blend_faces(face1, face2, alpha)
    new_face = blended_face[0]
    new_face_attr = blended_face[1]
    new_face_floue = apply_blur_around_face(new_face, new_face_attr, blur_strength=11)

    new_face_floue_crop = crop(new_face_floue)
    #new_face_floue_crop = draw_landmarks(new_face_floue_crop)
    return new_face_floue_crop

#####################################

#FONCTION QUI BLEND ALEAT UNE LIST D'IMAGE ET EN RENVOIE K
#####################################
def test_visage_non_detect(image):
    detector = dlib.get_frontal_face_detector()
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return False
    return True

def test_list_visage(list_visage):
    list_finale = []
    for visage in list_visage:
        if test_visage_non_detect(visage) : list_finale.append(visage)
    return list_finale

def apply_random_blending(faces, k):
    """
    Blend and blur random face combinations.
    If only two faces are provided, apply multiple alpha blending.
    If more, blend k random pairs and display them in a grid (auto-sized).
    Also saves the results as images.
    """
    faces = test_list_visage(faces)

    save_path = "/Users/ouiamelhajji/Documents/INSA/4A/S2/devweb/pièces_jointes"

    # Détermination automatique des lignes/colonnes selon k (max 3 colonnes)
    cols = 3
    rows = math.ceil(k / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)  # flatten even if 1D or 2D
    list_portrait=[]
    if len(faces) == 2:
        alphas = list(np.linspace(0.3, 0.6, k))
        for i, alpha in enumerate(alphas):
            blended_face = blend_faces(faces[0], faces[1], alpha=alpha)
            new_face = blended_face[0]
            new_face_attr = blended_face[1]
            new_face_floue = apply_blur_around_face(new_face, new_face_attr, blur_strength=11)

            axes[i].imshow(new_face_floue)
            axes[i].axis('off')

            file_path = os.path.join(save_path, f"blended_{i}.png")
            new_face_floue.save(file_path)
            list_portrait.append(new_face_floue)
    elif len(faces) > 2:
        for i in range(k):
            k_idx = random.randint(0, len(faces) - 1)
            s_idx = random.randint(0, len(faces) - 1)

            blended_face = blend_faces(faces[k_idx], faces[s_idx], alpha=0.45)
            new_face = blended_face[0]
            new_face_attr = blended_face[1]
            new_face_floue = apply_blur_around_face(new_face, new_face_attr, blur_strength=11)

            axes[i].imshow(new_face_floue)
            axes[i].axis('off')

            file_path = os.path.join(save_path, f"blended_{i}.png")
            new_face_floue.save(file_path)
            list_portrait.append(new_face_floue)
    # Cache les axes restants s’il y en a (ex : 5 images → 6 cases créées)
    for j in range(k, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    for face in list_portrait:
        face = face.resize((250, 250), Image.LANCZOS)

    return list_portrait


#####################################

#FONCTION QUI applique mutation ALEAT UNE LIST D'IMAGE ET la RENVOIE
#####################################

def mutation_aleatoire(image, proba=0.1):
    proba_rand = np.random.rand()
    if proba_rand>proba : return image
    attribut_rand = random.randint(1, 3)
    direction_rand = random.randint(1, 4)

    if attribut_rand==1 :
        attribut = "yeux"
        if direction_rand==1 :
            direction = "centre"
            new_image = modifier_landmarks(image, 2, attribut=attribut, direction=direction)
            return new_image
        elif direction_rand==2 :
            direction = "bord"
            new_image = modifier_landmarks(image, 2, attribut=attribut, direction=direction)
            return new_image
        elif direction_rand==3 :
            direction = "bas"
            new_image = modifier_landmarks(image, 2, attribut=attribut, direction=direction)
            return new_image
        elif direction_rand==4 :
            direction = "haut"
            new_image = modifier_landmarks(image, 2, attribut=attribut, direction=direction)
            return new_image

    if attribut_rand==2 :
        attribut = "bouche"
        if direction_rand==1 :
            direction = "droite"
            new_image = modifier_landmarks(image, 2, attribut=attribut, direction=direction)
            return new_image
        elif direction_rand==2 :
            direction = "gauche"
            new_image = modifier_landmarks(image, 2, attribut=attribut, direction=direction)
            return new_image
        elif direction_rand==3 :
            direction = "bas"
            new_image = modifier_landmarks(image, 2, attribut=attribut, direction=direction)
            return new_image
        elif direction_rand==4 :
            direction = "haut"
            new_image = modifier_landmarks(image, 2, attribut=attribut, direction=direction)
            return new_image

    if attribut_rand==3 :
        attribut = "nez"
        if direction_rand==1 :
            direction = "droite"
            new_image = modifier_landmarks(image, 2, attribut=attribut, direction=direction)
            return new_image
        elif direction_rand==2 :
            direction = "gauche"
            new_image = modifier_landmarks(image, 2, attribut=attribut, direction=direction)
            return new_image
        elif direction_rand==3 :
            direction = "bas"
            new_image = modifier_landmarks(image, 2, attribut=attribut, direction=direction)
            return new_image
        elif direction_rand==4 :
            direction = "haut"
            new_image = modifier_landmarks(image, 2, attribut=attribut, direction=direction)
            return new_image

#print(mutation_aleatoire(face1, proba=0.1))


def apply_random_modification(faces):
    faces_modifiées=[]
    for face in faces:
        faces_modifiées.append(mutation_aleatoire(face, proba=0.1))


    return faces_modifiées
#hihi

#print(apply_random_modification(faces))

#####################################

#RUN RUN

#####################################

""""
#face1 = Image.open("000004.jpg")
#face2 = Image.open("000001.jpg")
#print(face1)

#blended_face = blend_faces(image_1, image_2, alpha=0.45)
#print(blended_face)
new_face = blended_face[0]
new_face_attr = blended_face[1]
new_face_floue = apply_blur_around_face(new_face, new_face_attr, blur_strength=11)

plt.imshow(new_face_floue)
plt.axis('off')
plt.show()
"""