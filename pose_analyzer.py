"""
Módulo de Análise de Pose para Detecção de Tipo de Contato
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional

class PoseAnalyzer:
    """Analisa poses dos jogadores para identificar tipo de contato"""
    
    def __init__(self):
        """Inicializa o analisador de pose usando MediaPipe"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def analyze_pose(self, image: np.ndarray) -> Optional[Dict]:
        """
        Analisa pose em uma imagem
        
        Args:
            image: Imagem RGB do jogador
            
        Returns:
            Dict com landmarks da pose ou None se não detectado
        """
        # Converter BGR para RGB se necessário
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extrair coordenadas dos landmarks
        landmarks = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        return {
            'landmarks': landmarks,
            'raw_landmarks': results.pose_landmarks
        }
    
    def detect_action_type(self, pose_data: Dict) -> str:
        """
        Detecta tipo de ação baseado na pose
        
        Args:
            pose_data: Dados da pose do jogador
            
        Returns:
            Tipo de ação: 'kick', 'tackle', 'push', 'jump', 'normal'
        """
        if not pose_data or 'landmarks' not in pose_data:
            return 'unknown'
        
        landmarks = pose_data['landmarks']
        
        # Pontos-chave do MediaPipe
        # 23, 24: quadris
        # 25, 26: joelhos
        # 27, 28: tornozelos
        # 11, 12: ombros
        # 15, 16: punhos
        
        try:
            # Detectar chute: perna elevada
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Se tornozelo está acima do quadril, pode ser chute
            if (left_ankle['y'] < left_hip['y'] - 0.2 or 
                right_ankle['y'] < right_hip['y'] - 0.2):
                return 'kick'
            
            # Detectar entrada/tackle: corpo inclinado, pernas estendidas
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            avg_hip_y = (left_hip['y'] + right_hip['y']) / 2
            avg_knee_y = (left_knee['y'] + right_knee['y']) / 2
            
            # Corpo muito inclinado para frente
            if avg_knee_y < avg_hip_y - 0.15:
                return 'tackle'
            
            # Detectar empurrão: braços estendidos para frente
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            # Braços estendidos horizontalmente
            if (abs(left_wrist['y'] - left_shoulder['y']) < 0.1 or
                abs(right_wrist['y'] - right_shoulder['y']) < 0.1):
                return 'push'
            
            # Detectar pulo: ambos os pés elevados
            if (left_ankle['y'] < left_hip['y'] and 
                right_ankle['y'] < right_hip['y']):
                return 'jump'
            
            return 'normal'
            
        except (KeyError, TypeError):
            return 'unknown'
    
    def calculate_contact_intensity(self, pose1: Dict, pose2: Dict) -> float:
        """
        Calcula intensidade de contato entre dois jogadores
        
        Args:
            pose1: Pose do jogador 1
            pose2: Pose do jogador 2
            
        Returns:
            Score de intensidade (0-1)
        """
        if not pose1 or not pose2:
            return 0.0
        
        try:
            landmarks1 = pose1['landmarks']
            landmarks2 = pose2['landmarks']
            
            # Calcular distância entre pontos-chave
            distances = []
            key_points = [11, 12, 15, 16, 23, 24, 27, 28]  # ombros, punhos, quadris, tornozelos
            
            for point_idx in key_points:
                if point_idx in landmarks1 and point_idx in landmarks2:
                    p1 = landmarks1[point_idx]
                    p2 = landmarks2[point_idx]
                    
                    dist = np.sqrt((p1['x'] - p2['x'])**2 + 
                                  (p1['y'] - p2['y'])**2 + 
                                  (p1['z'] - p2['z'])**2)
                    distances.append(dist)
            
            if not distances:
                return 0.0
            
            # Quanto menor a distância, maior a intensidade
            avg_distance = np.mean(distances)
            intensity = max(0, 1 - avg_distance * 2)  # Normalizar
            
            return float(intensity)
            
        except (KeyError, TypeError):
            return 0.0
    
    def draw_pose(self, image: np.ndarray, pose_data: Dict) -> np.ndarray:
        """
        Desenha pose sobre a imagem
        
        Args:
            image: Imagem original
            pose_data: Dados da pose
            
        Returns:
            Imagem com pose desenhada
        """
        if not pose_data or 'raw_landmarks' not in pose_data:
            return image
        
        annotated_image = image.copy()
        self.mp_drawing.draw_landmarks(
            annotated_image,
            pose_data['raw_landmarks'],
            self.mp_pose.POSE_CONNECTIONS
        )
        
        return annotated_image
    
    def __del__(self):
        """Limpa recursos"""
        if hasattr(self, 'pose'):
            self.pose.close()

if __name__ == "__main__":
    # Teste básico
    analyzer = PoseAnalyzer()
    print("Analisador de pose inicializado com sucesso!")

