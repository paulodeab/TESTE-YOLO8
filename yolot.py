import cv2
from ultralytics import YOLO
import torch
import time

# Verificar se CUDA está disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Carregar o modelo YOLOv8-Seg pré-treinado
model = YOLO("yolov8n-seg.pt")  # Pode usar "yolov8s-seg.pt" para maior precisão

# Captura de vídeo
cap = cv2.VideoCapture(1)  # 0 para webcam padrão
if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

# Configurações da janela
cv2.namedWindow("YOLOv8 Segmentação em Tempo Real", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Segmentação em Tempo Real", 1024, 648)

# Loop de inferência em tempo real
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o quadro.")
        break

    # Redimensionar para 640x360 para acelerar processamento
    frame_resized = cv2.resize(frame, (640, 360))

    # Fazer inferência
    results = model.predict(frame_resized, device=device, conf=0.5, half=True, verbose=False)

    # Plotar os resultados diretamente no quadro
    annotated_frame = results[0].plot()

    # Calcular FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Mostrar o vídeo segmentado
    cv2.imshow("YOLOv8 Segmentação em Tempo Real", annotated_frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
