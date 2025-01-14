# import cv2
# from ultralytics import YOLO
# import numpy as np

# def run_segmentation(model_path, source=0):
#     """
#     Executa a segmentação em tempo real usando o modelo YOLOv8-Seg.

#     Args:
#         model_path (str): Caminho para o modelo YOLOv8-Seg treinado (ex: 'best.pt').
#         source (int or str): 0 para webcam, ou caminho para arquivo de vídeo.
#     """

#     # Carrega o modelo YOLOv8-Seg
#     model = YOLO(model_path)

#     # Abre a fonte de vídeo (webcam ou arquivo de vídeo)
#     cap = cv2.VideoCapture(source)

#     if not cap.isOpened():
#         print("Erro: Não foi possível abrir a fonte de vídeo.")
#         return

#     # Loop para capturar os frames do vídeo
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Erro: Não foi possível capturar o quadro.")
#             break

#         # Realiza a inferência
#         results = model(frame)

#         # Obtemos as máscaras do resultado
#         masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

#         # Criar uma máscara preta do mesmo tamanho do frame
#         mask_overlay = np.zeros_like(frame, dtype=np.uint8)

#         # Desenhar cada máscara com uma cor específica
#         for mask in masks:
#             # Converter a máscara em binária (0 ou 255)
#             mask_binary = (mask * 255).astype(np.uint8)

#             # Redimensiona a máscara binária para coincidir com o tamanho do frame
#             mask_binary_resized = cv2.resize(mask_binary, (frame.shape[1], frame.shape[0]))

#             # Se a máscara tiver 1 canal, converter para 3 canais
#             if len(mask_binary_resized.shape) == 2:
#                 mask_binary_resized = cv2.cvtColor(mask_binary_resized, cv2.COLOR_GRAY2BGR)

#             # Aplicar uma cor à máscara (por exemplo, azul)
#             color_mask = mask_binary_resized * np.array([255, 50, 0], dtype=np.uint8)

#             # Adicionar a máscara colorida ao overlay
#             mask_overlay = cv2.add(mask_overlay, color_mask)

#         # Mesclar a máscara colorida com o frame original
#         frame = cv2.addWeighted(frame, 0.6, mask_overlay, 0.4, 0)

#         # Exibe o frame anotado
#         cv2.imshow("Segmentação YOLOv8-Seg", frame)

#         # Pressione 'q' para sair
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Libera os recursos
#     cap.release()
#     cv2.destroyAllWindows()

# # Caminho para o modelo YOLOv8-Seg treinado
# model_path = "runs/segment/train/weights/best.pt"  # Substitua pelo caminho correto do seu modelo

# # Fonte de vídeo: 0 para webcam, ou caminho para um arquivo de vídeo
# source = 'video.mp4'  # Para webcam, ou 'caminho/para/video.mp4' para vídeo

# # Executa o teste de segmentação
# run_segmentation(model_path, source)

import cv2
from ultralytics import YOLO
import numpy as np

def run_segmentation(model_path, source=0, conf_threshold=0.88):
    """
    Executa a segmentação em tempo real usando o modelo YOLOv8-Seg com filtro de confiança.

    Args:
        model_path (str): Caminho para o modelo YOLOv8-Seg treinado (ex: 'best.pt').
        source (int or str): 0 para webcam, ou caminho para arquivo de vídeo.
        conf_threshold (float): Limiar de confiança para filtrar as detecções.
    """

    # Carrega o modelo YOLOv8-Seg
    model = YOLO(model_path)

    # Abre a fonte de vídeo (webcam ou arquivo de vídeo)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a fonte de vídeo.")
        return

    # Loop para capturar os frames do vídeo
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o quadro.")
            break

        # Realiza a inferência com o limiar de confiança especificado
        results = model(frame, conf=conf_threshold)

        # Obtemos as máscaras do resultado
        masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

        # Criar uma máscara preta do mesmo tamanho do frame
        mask_overlay = np.zeros_like(frame, dtype=np.uint8)

        # Desenhar cada máscara com uma cor específica
        for mask in masks:
            # Converter a máscara em binária (0 ou 255)
            mask_binary = (mask * 255).astype(np.uint8)

            # Redimensiona a máscara binária para coincidir com o tamanho do frame
            mask_binary_resized = cv2.resize(mask_binary, (frame.shape[1], frame.shape[0]))

            # Se a máscara tiver 1 canal, converter para 3 canais
            if len(mask_binary_resized.shape) == 2:
                mask_binary_resized = cv2.cvtColor(mask_binary_resized, cv2.COLOR_GRAY2BGR)

            # Aplicar uma cor à máscara (por exemplo, azul)
            color_mask = mask_binary_resized * np.array([255, 50, 0], dtype=np.uint8)

            # Adicionar a máscara colorida ao overlay
            mask_overlay = cv2.add(mask_overlay, color_mask)

        # Mesclar a máscara colorida com o frame original
        frame = cv2.addWeighted(frame, 0.6, mask_overlay, 0.4, 0)

        # Exibe o frame anotado
        cv2.imshow("Segmentação YOLOv8-Seg", frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera os recursos
    cap.release()
    cv2.destroyAllWindows()

# Caminho para o modelo YOLOv8-Seg treinado
model_path = "runs/segment/train/weights/best.pt"  # Substitua pelo caminho correto do seu modelo

# Fonte de vídeo: 0 para webcam, ou caminho para um arquivo de vídeo
source = 1  # Para webcam, ou 'caminho/para/video.mp4' para vídeo

# Executa o teste de segmentação
run_segmentation(model_path, source)

