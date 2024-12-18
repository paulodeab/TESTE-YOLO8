import os
import json
import glob

def convert_labelme_to_yolo_seg(labelme_dir, output_dir, class_name="glove"):
    """
    Converte arquivos JSON anotados no Labelme para o formato YOLOv8-Seg.

    Args:
        labelme_dir (str): Caminho para a pasta com arquivos JSON do Labelme.
        output_dir (str): Caminho para a pasta de saída dos arquivos YOLO.
        class_name (str): Nome da classe que será convertida.
    """
    os.makedirs(output_dir, exist_ok=True)  # Cria a pasta de saída
    class_id = 0  # Define o ID da classe (0 para "glove")

    # Processa todos os arquivos JSON
    for json_file in glob.glob(os.path.join(labelme_dir, "*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Nome do arquivo de saída no formato .txt
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(json_file))[0] + ".txt")

        with open(output_file, "w") as out:
            for shape in data["shapes"]:
                if shape["label"] == class_name:  # Verifica se a classe é a correta
                    points = shape["points"]  # Coordenadas do polígono
                    h, w = data["imageHeight"], data["imageWidth"]  # Dimensões da imagem

                    # Normaliza as coordenadas (0 a 1)
                    normalized_points = [(x / w, y / h) for x, y in points]
                    coords = [f"{x:.6f} {y:.6f}" for x, y in normalized_points]

                    # Escreve no formato YOLOv8
                    out.write(f"{class_id} " + " ".join(coords) + "\n")

        print(f"Convertido: {json_file} -> {output_file}")

    print("Conversão concluída! Arquivos YOLOv8-Seg salvos em:", output_dir)


# Caminhos para os diretórios
labelme_dir = "C:\\Users\\pmariano.FIEMG\\Pictures\\Camera Roll\\label - backup"  # Substitua pelo caminho da pasta JSON
output_dir = "C:\\Users\\pmariano.FIEMG\\Pictures\\Camera Roll\\label - yolo"    # Substitua pelo caminho da saída

# Executa a conversão
convert_labelme_to_yolo_seg(labelme_dir, output_dir, class_name="glove")




