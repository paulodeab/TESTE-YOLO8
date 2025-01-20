import cv2
from ultralytics import YOLO

def run_segmentation(model_path, source=0):
    """
    Executa a segmentação em tempo real usando o modelo YOLOv8-Seg.

    Args:
        model_path (str): Caminho para o modelo YOLOv8-Seg treinado (ex: 'best.pt').
        source (int or str): 0 para webcam, ou caminho para arquivo de vídeo.
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
        

        # Desenha os resultados da segmentação no frame
        annotated_frame = results[0].plot()

        
        # Exibe o frame anotado
        cv2.imshow("Segmentação YOLOv8-Seg", annotated_frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera os recursos
    cap.release()
    cv2.destroyAllWindows()

# Caminho para o modelo YOLOv8-Seg treinado
model_path = "runs/segment/train2/weights/best2.pt"  # Substitua pelo caminho correto do seu modelo

# Fonte de vídeo: 0 para webcam, ou caminho para um arquivo de vídeo
source = 'v1.mp4'  # Para webcam, ou 'caminho/para/video.mp4' para vídeo

# Executa o teste de segmentação
run_segmentation(model_path, source)
