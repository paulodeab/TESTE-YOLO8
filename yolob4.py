import cv2
import numpy as np
from ultralytics import YOLO

def run_segmentation(model_path, source=0, confidence_threshold=0.75):
    """
    Executa a segmentação em tempo real usando o modelo YOLOv8-Seg.

    Args:
        model_path (str): Caminho para o modelo YOLOv8-Seg treinado (ex: 'best.pt').
        source (int or str): 0 para webcam, ou caminho para arquivo de vídeo.
        confidence_threshold (float): Limite mínimo de confiança para exibir as segmentações.
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

        # Realiza a inferência
        results = model(frame)

        # Processa os resultados
        for result in results:
            masks = result.masks
            if masks is not None:
                for mask, conf in zip(masks.data, result.boxes.conf):
                    if conf > confidence_threshold:
                        # Converte a máscara para um formato adequado para OpenCV
                        mask = mask.cpu().numpy().astype(np.uint8) * 255
                        
                        # Encontra os contornos da máscara
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Desenha os contornos no frame
                        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)  # Verde

                        # Exibe o valor de confiança próximo aos contornos
                        x, y, w, h = cv2.boundingRect(mask)
                        cv2.putText(frame, f"{conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
source = 'video.mp4'  # Para webcam, ou 'caminho/para/video.mp4' para vídeo

# Executa o teste de segmentação
run_segmentation(model_path, source)