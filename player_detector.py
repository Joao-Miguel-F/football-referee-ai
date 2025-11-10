"""
Módulo de Detecção de Jogadores e Bola usando YOLOv8
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict

class PlayerDetector:
    """Detecta jogadores, bola e árbitros em frames de vídeo de futebol"""
    
    def __init__(self, model_name='yolov8n.pt'):
        """
        Inicializa o detector YOLO
        
        Args:
            model_name: Nome do modelo YOLO (yolov8n.pt, yolov8s.pt, etc.)
        """
        print(f"Carregando modelo YOLO: {model_name}")
        self.model = YOLO(model_name)
        
        # Classes relevantes do COCO dataset
        self.person_class = 0  # pessoa
        self.sports_ball_class = 32  # bola esportiva
        
    def detect_frame(self, frame: np.ndarray, conf_threshold: float = 0.3) -> Dict:
        """
        Detecta objetos em um frame
        
        Args:
            frame: Frame de vídeo (numpy array)
            conf_threshold: Limiar de confiança para detecções
            
        Returns:
            Dict com detecções: {
                'players': [(x1, y1, x2, y2, conf), ...],
                'ball': [(x1, y1, x2, y2, conf), ...],
                'frame_annotated': frame com anotações
            }
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        players = []
        ball = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                if cls == self.person_class:
                    players.append((int(x1), int(y1), int(x2), int(y2), conf))
                elif cls == self.sports_ball_class:
                    ball.append((int(x1), int(y1), int(x2), int(y2), conf))
        
        # Frame anotado
        frame_annotated = results[0].plot() if results else frame.copy()
        
        return {
            'players': players,
            'ball': ball,
            'frame_annotated': frame_annotated
        }
    
    def extract_player_regions(self, frame: np.ndarray, players: List[Tuple]) -> List[np.ndarray]:
        """
        Extrai regiões de imagem dos jogadores detectados
        
        Args:
            frame: Frame original
            players: Lista de bounding boxes dos jogadores
            
        Returns:
            Lista de imagens recortadas dos jogadores
        """
        player_crops = []
        for x1, y1, x2, y2, conf in players:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                player_crops.append(crop)
        return player_crops
    
    def get_interaction_region(self, frame: np.ndarray, players: List[Tuple], 
                               expansion_factor: float = 1.5) -> np.ndarray:
        """
        Extrai região de interação entre jogadores próximos
        
        Args:
            frame: Frame original
            players: Lista de jogadores detectados
            expansion_factor: Fator de expansão da região
            
        Returns:
            Região de imagem contendo interação
        """
        if len(players) < 2:
            return frame
        
        # Encontrar bounding box que engloba todos os jogadores
        all_x1 = min([p[0] for p in players])
        all_y1 = min([p[1] for p in players])
        all_x2 = max([p[2] for p in players])
        all_y2 = max([p[3] for p in players])
        
        # Expandir região
        width = all_x2 - all_x1
        height = all_y2 - all_y1
        center_x = (all_x1 + all_x2) // 2
        center_y = (all_y1 + all_y2) // 2
        
        new_width = int(width * expansion_factor)
        new_height = int(height * expansion_factor)
        
        x1 = max(0, center_x - new_width // 2)
        y1 = max(0, center_y - new_height // 2)
        x2 = min(frame.shape[1], center_x + new_width // 2)
        y2 = min(frame.shape[0], center_y + new_height // 2)
        
        return frame[y1:y2, x1:x2]
    
    def detect_close_players(self, players: List[Tuple], 
                            distance_threshold: float = 100) -> List[List[int]]:
        """
        Detecta grupos de jogadores próximos (possível interação/falta)
        
        Args:
            players: Lista de jogadores detectados
            distance_threshold: Distância máxima em pixels para considerar próximos
            
        Returns:
            Lista de grupos de índices de jogadores próximos
        """
        if len(players) < 2:
            return []
        
        # Calcular centros dos jogadores
        centers = []
        for x1, y1, x2, y2, conf in players:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y))
        
        # Encontrar jogadores próximos
        close_groups = []
        for i in range(len(centers)):
            group = [i]
            for j in range(i + 1, len(centers)):
                dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                              (centers[i][1] - centers[j][1])**2)
                if dist < distance_threshold:
                    group.append(j)
            
            if len(group) > 1:
                close_groups.append(group)
        
        return close_groups

if __name__ == "__main__":
    # Teste básico
    detector = PlayerDetector()
    print("Detector de jogadores inicializado com sucesso!")

