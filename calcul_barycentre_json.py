import cv2
import numpy as np
import os
import argparse
import json

tous_les_barycentres = {}
frame_counter = 1

def traiter_tous_les_dossiers(parent_folder):
    """
    Parcourt tous les sous-dossiers du dossier parent et applique la fonction calcul_barycentres
    à chaque sous-dossier contenant une image et des masques.
    """
    # Vérifier que le dossier parent existe
    if not os.path.exists(parent_folder):
        raise ValueError(f"Le dossier {parent_folder} n'existe pas !")

    # Lister tous les sous-dossiers
    for subfolder in sorted(os.listdir(parent_folder)):  # Trier pour un traitement ordonné
        subfolder_path = os.path.join(parent_folder, subfolder)
        
        if os.path.isdir(subfolder_path):  # Vérifier que c'est bien un dossier
            # Chemin de l'image
            image_path = os.path.join(subfolder_path, "segmented_image.jpg")
            masks_folder_path = subfolder_path  # Les masques sont dans le même dossier

            # Vérifier si l'image et au moins un masque existent
            if os.path.exists(image_path) and masks_folder_path:
                print(f"📂 Traitement de {subfolder_path}...")
                
                # Appeler la fonction avec ce dossier de sortie spécifique
                calcul_barycentres(image_path, masks_folder_path)
            else:
                print(f"⚠️ Aucun masque trouvé ou image manquante dans {subfolder_path}. Skipping.")
    
    # Sauvegarder tous les barycentres dans un fichier JSON général
    output_json_general = os.path.join(parent_folder, "barycentres_general.json")
    with open(output_json_general, "w") as f:
        json.dump(tous_les_barycentres, f, indent=4, sort_keys=True)  # Trier les clés avant d'écrire
    print(f"✅ Tous les barycentres enregistrés dans {output_json_general}")

def calcul_barycentres(image_path, masks_folder_path, min_area=500):
    global frame_counter  # Utiliser le compteur global
    global tous_les_barycentres  # Utiliser le dictionnaire global pour tous les barycentres

    # Charger l'image originale
    image_before = cv2.imread(image_path)
    if image_before is None:
        raise ValueError(f"Erreur : Impossible de charger l'image {image_path}")

    # Lister tous les fichiers du dossier des masques
    mask_files = [f for f in os.listdir(masks_folder_path) if f.startswith("mask_") and f.endswith(".png")]

    if not mask_files:
        raise ValueError(f"Aucun fichier de masque trouvé dans le dossier {masks_folder_path}")

    # Générer le nouveau nom de l'image (frame_xxxxx.jpg)
    frame_name = f"frame_{frame_counter:05d}.jpg"  # Format : frame_00001.jpg, frame_00002.jpg, etc.
    tous_les_barycentres[frame_name] = {}

    # Traiter chaque fichier de masque
    for mask_file in mask_files:
        mask_path = os.path.join(masks_folder_path, mask_file)

        # Charger le masque binaire
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Erreur : Impossible de charger le masque {mask_path}")
            continue

        # Appliquer l'ouverture morphologique pour enlever les petits points
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Appliquer un filtre médian pour lisser le masque
        mask = cv2.medianBlur(mask, 5)  # Taille du noyau 5x5

        # Trouver les contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrer les petits objets
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if filtered_contours:  # Vérifier s'il reste des contours après le filtrage
            # Garder seulement le plus grand contour
            largest_contour = max(filtered_contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:  # Vérifier que le contour n'est pas vide
                x_bary = int(M["m10"] / M["m00"])
                y_bary = int(M["m01"] / M["m00"])

                # Vérifier si le barycentre est dans le masque
                inside = cv2.pointPolygonTest(largest_contour, (x_bary, y_bary), False)

                if inside < 0:  # Si le barycentre est en dehors
                    print(f"⚠️ Barycentre en dehors du masque ! Correction pour {mask_file}")

                    # Trouver le point du contour le plus proche du barycentre
                    min_dist = float("inf")
                    closest_point = (x_bary, y_bary)

                    for point in largest_contour:
                        px, py = point[0]
                        dist = np.linalg.norm(np.array([px, py]) - np.array([x_bary, y_bary]))
                        if dist < min_dist:
                            min_dist = dist
                            closest_point = (px, py)

                    # Mettre à jour le barycentre avec le point le plus proche sur le contour
                    x_bary, y_bary = closest_point
                    print(f"✅ Barycentre corrigé en ({x_bary}, {y_bary})")

                # Ajouter le barycentre au dictionnaire général
                tous_les_barycentres[frame_name][mask_file] = [int(x_bary), int(y_bary)]
                cv2.circle(image_before, (x_bary, y_bary), 5, (255, 255, 255), -1)  # Dessiner le barycentre

    # Incrémenter le compteur de frames
    frame_counter += 1

# Exemple d'utilisation avec argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traitement des images et des masques.")
    parser.add_argument("--parent-folder", type=str, help="Le chemin du dossier parent contenant les sous-dossiers avec images et masques.")
    
    # Parser les arguments
    args = parser.parse_args()
    
    # Appeler la fonction avec le dossier parent passé en argument
    traiter_tous_les_dossiers(args.parent_folder)
