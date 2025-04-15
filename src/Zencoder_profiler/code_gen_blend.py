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
import os
import math


def align_face(img, landmarks_init, landmarks_fin, size=(256, 256)):
    """Aligne un visage en utilisant une transformation homographique basée sur les 68 landmarks."""
    if isinstance(img, Image.Image):
        img = np.array(img)
    init_points = np.float32(landmarks_init)
    fin_points = np.float32(landmarks_fin)
    matrix, _ = cv2.findHomography(init_points, fin_points)
    aligned_face = cv2.warpPerspective(img, matrix, size)
    return aligned_face

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

def draw_landmarks(image):
    """Dessine les 68 points caractéristiques sur l'image."""
    image = np.array(image)
    landmarks = get_landmarks(image)
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    return image

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

def find_closest_landmark(landmarks, point):
    """Trouve l'index du landmark le plus proche d'un point donné."""
    landmarks = np.array(landmarks)
    distances = np.linalg.norm(landmarks - np.array(point), axis=1)  # Distance Euclidienne
    return np.argmin(distances)  # Index du plus proche

def ajuster_visage(face, landmarks_init, landmarks_fin):
    """Applique une déformation locale avec interpolation triangulaire tout en conservant le reste de l'image."""
    if landmarks_init is None or landmarks_fin is None:
        print("Erreur : Impossible de réaliser l'ajustement du visage.")
        print("Il manque des landmarks.")
        return None

    if isinstance(face, Image.Image):
        face = np.array(face)
    
    h, w = face.shape[:2]
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)

    for lm in landmarks_init:
        subdiv.insert((float(lm[0]), float(lm[1])))  
    triangles = subdiv.getTriangleList().astype(int)
    masque_visage = np.zeros((h, w), dtype=np.uint8)
    new_face = np.zeros_like(face)

    for tri in triangles:
        x1, y1, x2, y2, x3, y3 = tri
        idx1 = find_closest_landmark(landmarks_init, (x1, y1))
        idx2 = find_closest_landmark(landmarks_init, (x2, y2))
        idx3 = find_closest_landmark(landmarks_init, (x3, y3))
        pts1 = np.array([landmarks_init[idx1], landmarks_init[idx2], landmarks_init[idx3]], np.float32)
        pts2 = np.array([landmarks_fin[idx1], landmarks_fin[idx2], landmarks_fin[idx3]], np.float32)

        M = cv2.getAffineTransform(pts1, pts2)
        warped_triangle = cv2.warpAffine(face, M, (w, h))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32([pts2]), 255)
        masque_visage = cv2.bitwise_or(masque_visage, mask)
        new_face = cv2.bitwise_or(new_face, cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask))

    face_sans_visage = cv2.bitwise_and(face, face, mask=cv2.bitwise_not(masque_visage))
    image_finale = cv2.bitwise_or(face_sans_visage, new_face)

    return Image.fromarray(image_finale)

def modifier_landmarks(face, valeur, attribut="yeux", direction="centre"):
    """
    Modifie les landmarks :
    oeil gauche : 36-41
    oeil droit : 42-47
    bouche : 48-68
    nez : 27-35
    contour du visage : 0-16

    Renvoie le visage déformé.
    """
    landmarks = get_landmarks(face)
    new_landmarks = landmarks.copy()
    if attribut=="yeux":
        oeil_gauche = list(range(36, 42))
        oeil_droit = list(range(42, 48))
        if direction=="centre":
            new_landmarks[oeil_gauche] +=[valeur, 0]
            new_landmarks[oeil_droit] +=[-valeur, 0]
        if direction=="bord":
            new_landmarks[oeil_gauche] +=[-valeur, 0]
            new_landmarks[oeil_droit] +=[valeur, 0]
        if direction=="bas":
            new_landmarks[oeil_gauche] +=[0, -valeur]
            new_landmarks[oeil_droit] +=[0, -valeur]
        if direction=="haut":
            new_landmarks[oeil_gauche] +=[0, valeur]
            new_landmarks[oeil_droit] +=[0, valeur]
            
        return ajuster_visage(face, landmarks, new_landmarks)
        
    if attribut=="bouche":
        bouche = list(range(48, 68))
        if direction=="droite":
            new_landmarks[bouche] +=[valeur, 0]
        if direction=="gauche":
            new_landmarks[bouche] +=[-valeur, 0]
        if direction=="bas":
            new_landmarks[bouche] +=[0, valeur]
        if direction=="haut":
            new_landmarks[bouche] +=[0, -valeur]
        return ajuster_visage(face, landmarks, new_landmarks)
        
    if attribut=="nez":
        nez = list(range(27, 36))
        if direction=="droite":
            new_landmarks[nez] +=[valeur, 0]
        if direction=="gauche":
            new_landmarks[nez] +=[-valeur, 0]
        if direction=="bas":
            new_landmarks[nez] +=[0, valeur]
        if direction=="haut":
            new_landmarks[nez] +=[0, -valeur]
        return ajuster_visage(face, landmarks, new_landmarks)
    return

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
    for face in faces:
        face = face.resize((250, 250), Image.LANCZOS)
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

def mutation_aleatoire(image, proba=0.2):
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


def apply_random_modification(faces):
    faces_modifiées=[]
    for face in faces:
        faces_modifiées.append(mutation_aleatoire(face, proba=0.1))


    return faces_modifiées

def apply_random_blending(faces, k):
    """
    Blend and blur random face combinations.
    If only two faces are provided, apply multiple alpha blending.
    If more, blend k random pairs and display them in a grid (auto-sized).
    Also saves the results as images.
    """
    faces = test_list_visage(faces)

    # Détermination automatique des lignes/colonnes selon k (max 3 colonnes)
    cols = 3
    rows = math.ceil(k / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)  # flatten even if 1D or 2D
    list_portrait = []

    if len(faces) == 2:
        alphas = list(np.linspace(0.3, 0.6, k))
        for i, alpha in enumerate(alphas): 
            blended_face = blend_faces(faces[0], faces[1], alpha=alpha)
            new_face = blended_face[0]
            new_face_attr = blended_face[1]
            new_face_floue = apply_blur_around_face(new_face, new_face_attr, blur_strength=11)

            axes[i].imshow(new_face_floue)
            axes[i].axis('off')

            list_portrait.append(new_face_floue)

    elif len(faces) > 2:
        i = 0
        seen_ids = set()
        while i < k:
            k_idx = random.randint(0, len(faces) - 1)
            s_idx = random.randint(0, len(faces) - 1)

            # Boucle jusqu'à ce qu'on ait un index différent
            while s_idx == k_idx:
                s_idx = random.randint(0, len(faces) - 1)

            blended_face = blend_faces(faces[k_idx], faces[s_idx], alpha=0.45)
            new_face = blended_face[0]
            new_face_attr = blended_face[1]
            new_face_floue = apply_blur_around_face(new_face, new_face_attr, blur_strength=11)

            # Vérification de similarité : 3 derniers caractères de l'ID
            hex_id = hex(id(new_face_floue))
            short_id = hex_id[-3:]

            if short_id in seen_ids:
                continue  # Trop similaire → skip

            seen_ids.add(short_id)

            axes[i].imshow(new_face_floue)
            axes[i].axis('off')

            list_portrait.append(new_face_floue)
            i += 1

    # Cache les axes restants s’il y en a (ex : 5 images → 6 cases créées)
    for j in range(k, len(axes)):
        axes[j].axis('off')
        
    return list_portrait


if __name__ == "__main__":
    
    list_attr=pd.read_csv("list_attr_celeba.txt" , sep=r"\s+", header = 0)
    pos_attr=pd.read_csv("list_landmarks_align_celeba.txt" , sep=r"\s+", header = 0)


    #image_1 = Image.open("img_align_celeba/000001.jpg")
    #image_2 = Image.open("img_align_celeba/000002.jpg")
    #image_3 = Image.open("img_align_celeba/000003.jpg")
    #image_4 = Image.open("img_align_celeba/000004.jpg")
    #image_5 = Image.open("img_align_celeba/000005.jpg")

    #PR RUN SUR ORDI OUIAM
    #PR RUN SUR ORDI OUIAM

    image_1 = Image.open("./000001.jpg")
    image_2 = Image.open("./000002.jpg")
    image_3 = Image.open("./000003.jpg")
    image_4 = Image.open("./000004.jpg")
    image_5 = Image.open("./000005.jpg")
    image_10 = Image.open("./000010.jpg")



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
    print(get_landmarks(image_1))
    print(blend_faces(image_1, image_2, alpha=0.45))
    #plt.imshow(image_1)
    #plt.scatter(x1, y2, c='red', marker='o', s=6)  # Points en rouge
    #plt.imshow(image_2)
    #plt.scatter(x2, y2, c='red', marker='o', s=6)  # Points en rouge
    plt.imshow(image_3)
    plt.scatter(x3, y3, c='red', marker='o', s=6)  # Points en rouge
    #plt.imshow(image_4)
    #plt.scatter(x4, y4, c='red', marker='o', s=6)  # Points en rouge
    plt.axis("on")  # Garde les axes activés (ou plt.axis("off") pour masquer)

    # Afficher le tout

    #####################################

    #RUN RUN

    #####################################


    face1 = Image.open("000004.jpg")
    face2 = Image.open("000001.jpg")
    #print(face1)
    

    blended_face = blend_faces(image_1, image_2, alpha=0.45)
    #print(blended_face)
    new_face = blended_face[0]
    new_face_attr = blended_face[1]
    new_face_floue = apply_blur_around_face(new_face, new_face_attr, blur_strength=11)

    plt.imshow(new_face_floue)
    plt.axis('off')
    plt.show()