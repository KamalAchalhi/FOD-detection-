import os
import cv2
import torch
import gc
import argparse
import numpy as np
import supervision as sv
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Libérer la mémoire GPU
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
gc.collect()

# Détection du périphérique
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

# Chemins vers les fichiers du modèle
CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Chargement du modèle SAM2
sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)
if num_gpus > 1:
    sam2_model = torch.nn.DataParallel(sam2_model)

# === ARGUMENTS CLI ===
parser = argparse.ArgumentParser(description="Segmentation d'un dossier d'images avec SAM2")
parser.add_argument("--input_dir", required=True, help="Dossier contenant les images à segmenter")
parser.add_argument("--output_dir", required=True, help="Dossier où stocker les résultats")
args = parser.parse_args()

# Vérifier que le dossier d'entrée existe
if not os.path.exists(args.input_dir):
    raise ValueError(f"Dossier introuvable : {args.input_dir}")

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(args.output_dir, exist_ok=True)

# Liste des images du dossier
image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    raise ValueError(f"Aucune image trouvée dans {args.input_dir}")

print(f"🔍 {len(image_files)} images trouvées dans {args.input_dir}. Début du traitement...")

# Création du générateur de masques avec des ajustements pour mieux détecter les objets complexes
mask_generator_advanced = SAM2AutomaticMaskGenerator(
    model=sam2_model.module if isinstance(sam2_model, torch.nn.DataParallel) else sam2_model,
    points_per_side=72,  # Légèrement augmenté pour améliorer la détection des objets complexes
    points_per_batch=256,
    pred_iou_thresh=0.72,  # Rend le modèle plus permissif pour capturer des objets plus complexes
    stability_score_thresh=0.88,  # Réduit légèrement pour capturer des objets plus difficiles
    stability_score_offset=0.55,
    mask_threshold=0.48,  # Réduit légèrement pour capturer plus de détails dans les zones complexes
    box_nms_thresh=0.40,  # Réduit pour éviter de filtrer des objets segmentés plus petits
    crop_n_layers=1,  # Active un niveau de recadrage pour améliorer la segmentation des objets complexes
    crop_nms_thresh=0.55,
    crop_overlap_ratio=0.55,
    min_mask_region_area=450  # Diminue légèrement pour inclure plus de petits objets détaillés
)

# Traitement des images
for image_name in tqdm(image_files, desc="Traitement des images", unit="img"):
    try:
        print(f"Traitement de {image_name}...")
        image_path = os.path.join(args.input_dir, image_name)
        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            print(f"⚠️ Erreur : Impossible de lire {image_name}, passage à l'image suivante.")
            continue

        # Redimensionner en conservant un ratio 16:9 pour éviter l'erreur OOM
        target_width = 1024
        target_height = int(target_width * 9 / 16)
        image_bgr = cv2.resize(image_bgr, (target_width, target_height))

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        print(f"🔹 Génération du masque pour {image_name}...")

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            sam2_result_advanced = mask_generator_advanced.generate(image_rgb)

        if not sam2_result_advanced:
            print(f"⚠️ Aucun masque détecté pour {image_name}. Skipping...")
            continue

        # Assurer que le dossier de sortie de l'image existe
        image_output_dir = os.path.join(args.output_dir, os.path.splitext(image_name)[0])
        os.makedirs(image_output_dir, exist_ok=True)

        # Sauvegarde des masques générés
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(sam_result=sam2_result_advanced)
        annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        segmented_image_path = os.path.join(image_output_dir, "segmented_image.jpg")
        cv2.imwrite(segmented_image_path, annotated_image)

        for i, mask_data in enumerate(sam2_result_advanced):
            mask = mask_data["segmentation"]
            mask_path = os.path.join(image_output_dir, f"mask_{i:05d}.png")
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

        print(f"✅ {image_name} segmentée avec succès !")

    except Exception as e:
        print(f"❌ Erreur lors du traitement de {image_name} : {str(e)}")
        continue

