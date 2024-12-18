import cv2
from ultralytics import YOLO
import numpy as np

# Carrega o modelo YOLOv8-Seg
model = YOLO('yolov8n-seg.pt')  # Use 'yolov8s-seg.pt' para mais precisão

# Inicializa a captura de vídeo (0 para webcam padrão)
cap = cv2.VideoCapture(1)

# Define a resolução da webcam (opcional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    # Captura frame a frame
    success, frame = cap.read()
    if not success:
        print("Falha ao capturar o vídeo.")
        break

    # Faz a inferência no frame atual
    results = model(frame, conf=0.5)

    # Filtra os resultados para a classe "person"
    for r in results:
        if r.masks and r.boxes:
            for mask, box in zip(r.masks.data, r.boxes.cls):
                # Verifica se a classe detectada é "person" (ID 0 no COCO dataset)
                if int(box) == 0:
                    # Converte a máscara para o formato OpenCV
                    mask = mask.cpu().numpy().astype('uint8') * 255
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                    # Encontra e desenha os contornos da máscara
                    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)  # Cor verde para os contornos

    # Exibe o frame com as máscaras desenhadas
    cv2.imshow('YOLOv8-Seg Real-Time (Somente Person)', frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
