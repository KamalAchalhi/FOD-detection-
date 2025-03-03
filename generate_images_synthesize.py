import json
import numpy as np
import xml.etree.ElementTree as ET
import argparse

def parse_metashape_xml(xml_file):
    """Extrait les matrices 4x4 des caméras depuis un fichier XML de Metashape."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    cameras = []
    for camera in root.findall(".//camera"):
        transform = camera.find("transform")
        if transform is not None:
            matrix = [float(x) for x in transform.text.split(" ")]
            matrix = np.array(matrix).reshape(4, 4)
            cameras.append(matrix)
    return cameras

def xml2json(dataparser_matrix, xml_Matrix, scale_factor):
    """Applique la transformation de Metashape vers NeRF et ajoute une ligne [0,0,0,1]."""
    json_matrix = dataparser_matrix @ xml_Matrix
    json_matrix[:, 3] *= scale_factor
    json_matrix[:, 1] *= -1
    json_matrix[:, 2] *= -1

    # Ajouter la ligne [0, 0, 0, 1] pour assurer une matrice 4x4
    json_matrix = np.vstack([json_matrix, np.array([0, 0, 0, 1])])

    return json_matrix

def main(xml_file, json_file):
    # Charger la transformation depuis le fichier JSON
    with open(json_file, 'r') as file:
        data = json.load(file)

    dataparser_matrix = np.array(data.get('transform')).reshape(3, 4)
    scale_factor = data.get('scale')

    # Extraire les matrices des fichiers XML
    fod_matrices = parse_metashape_xml(xml_file)

    # Construire la structure JSON finale
    output_data = {
        "default_fov": 25.0,
        "default_transition_sec": 2.0,
        "keyframes": [],
        "camera_type": "perspective",
        "render_height": 4024.0,
        "render_width": 6048.0,
        "fps": 30.0,
        "seconds": 10.0,
        "is_cycle": False,
        "smoothness_value": 0.0,
        "camera_path": []
    }

    # Remplir le champ "camera_path"
    for i, fod_matrix in enumerate(fod_matrices):
        json_matrix = xml2json(dataparser_matrix, fod_matrix, scale_factor)
        
        camera_entry = {
            "camera_to_world": json_matrix.flatten().tolist(),
            "fov": 25.0,
            "aspect": 0.6653439153439153
        }
        
        output_data["camera_path"].append(camera_entry)

    # Sauvegarde du JSON dans un fichier de sortie
    output_filename = "output_camera_path.json"
    with open(output_filename, "w") as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"Fichier JSON '{output_filename}' généré avec succès !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertit un fichier XML Metashape en JSON compatible NeRF.")
    parser.add_argument("xml_file", help="Fichier XML extrait de Metashape")
    parser.add_argument("json_file", help="Fichier JSON contenant la transformation et le scale factor")
    
    args = parser.parse_args()
    main(args.xml_file, args.json_file)
