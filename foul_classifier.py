"""
Módulo de Classificação de Faltas usando CNN
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from typing import Dict, Tuple

class FoulClassifier:
    """Classificador de faltas baseado em CNN"""
    
    def __init__(self, model_type='mobilenet', num_classes=3):
        """
        Inicializa o classificador
        
        Args:
            model_type: Tipo de modelo ('mobilenet', 'resnet', 'efficientnet')
            num_classes: Número de classes (3: sem falta, falta leve, falta grave)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.model_type = model_type
        
        # Carregar modelo pré-treinado
        self.model = self._load_model(model_type, num_classes)
        self.model.to(self.device)
        self.model.eval()
        
        # Transformações para imagens
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Mapeamento de classes
        self.class_names = {
            0: 'no_foul',
            1: 'minor_foul',
            2: 'serious_foul'
        }
        
    def _load_model(self, model_type: str, num_classes: int) -> nn.Module:
        """
        Carrega modelo pré-treinado e adapta para classificação de faltas
        
        Args:
            model_type: Tipo de modelo
            num_classes: Número de classes
            
        Returns:
            Modelo PyTorch
        """
        if model_type == 'mobilenet':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
            
        elif model_type == 'resnet':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        elif model_type == 'efficientnet':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            
        else:
            raise ValueError(f"Modelo não suportado: {model_type}")
        
        return model
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Pré-processa frame para entrada no modelo
        
        Args:
            frame: Frame de vídeo (BGR)
            
        Returns:
            Tensor processado
        """
        # Converter BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Aplicar transformações
        tensor = self.transform(frame_rgb)
        
        # Adicionar dimensão de batch
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def classify_frame(self, frame: np.ndarray) -> Dict:
        """
        Classifica um frame
        
        Args:
            frame: Frame de vídeo
            
        Returns:
            Dict com resultado da classificação
        """
        # Pré-processar
        input_tensor = self.preprocess_frame(frame)
        
        # Inferência
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        
        return {
            'class': self.class_names[predicted_class],
            'class_id': predicted_class,
            'confidence': confidence_score,
            'probabilities': {
                self.class_names[i]: probabilities[0][i].item() 
                for i in range(self.num_classes)
            }
        }
    
    def classify_sequence(self, frames: list) -> Dict:
        """
        Classifica sequência de frames (análise temporal)
        
        Args:
            frames: Lista de frames
            
        Returns:
            Dict com classificação agregada
        """
        if not frames:
            return {'class': 'unknown', 'confidence': 0.0}
        
        # Classificar cada frame
        classifications = []
        for frame in frames:
            result = self.classify_frame(frame)
            classifications.append(result)
        
        # Agregar resultados (votação por maioria ponderada por confiança)
        class_votes = {name: 0.0 for name in self.class_names.values()}
        
        for result in classifications:
            class_votes[result['class']] += result['confidence']
        
        # Classe com maior voto ponderado
        best_class = max(class_votes, key=class_votes.get)
        avg_confidence = class_votes[best_class] / len(classifications)
        
        return {
            'class': best_class,
            'confidence': avg_confidence,
            'frame_count': len(frames),
            'individual_results': classifications
        }
    
    def get_activation_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Gera mapa de ativação (GradCAM-like) para explicabilidade
        
        Args:
            frame: Frame de vídeo
            
        Returns:
            Mapa de calor sobreposto ao frame
        """
        # Simplificação: retorna frame original
        # Em produção, implementar GradCAM++ real
        return frame

class FoulSeverityAnalyzer:
    """Analisa severidade da falta para determinar cartões"""
    
    def __init__(self):
        self.severity_thresholds = {
            'yellow_card': 0.6,  # Falta imprudente
            'red_card': 0.85     # Falta com força excessiva
        }
    
    def analyze_severity(self, 
                        foul_classification: Dict,
                        contact_intensity: float,
                        action_type: str,
                        is_penalty_area: bool = False) -> Dict:
        """
        Analisa severidade da falta e determina cartão
        
        Args:
            foul_classification: Resultado do classificador de faltas
            contact_intensity: Intensidade do contato (0-1)
            action_type: Tipo de ação detectada
            is_penalty_area: Se ocorreu na área penal
            
        Returns:
            Dict com análise de severidade
        """
        severity_score = 0.0
        card = 'none'
        reasoning = []
        
        # Análise baseada na classificação
        if foul_classification['class'] == 'serious_foul':
            severity_score += 0.7
            reasoning.append("Falta grave detectada pelo classificador")
        elif foul_classification['class'] == 'minor_foul':
            severity_score += 0.4
            reasoning.append("Falta leve detectada")
        
        # Análise baseada na intensidade do contato
        severity_score += contact_intensity * 0.3
        if contact_intensity > 0.7:
            reasoning.append(f"Contato de alta intensidade ({contact_intensity:.2f})")
        
        # Análise baseada no tipo de ação
        action_severity = {
            'kick': 0.5,
            'tackle': 0.6,
            'push': 0.4,
            'jump': 0.3,
            'normal': 0.0
        }
        severity_score += action_severity.get(action_type, 0.0) * 0.2
        
        # Determinar cartão
        if severity_score >= self.severity_thresholds['red_card']:
            card = 'red'
            reasoning.append("Força excessiva - Cartão Vermelho")
        elif severity_score >= self.severity_thresholds['yellow_card']:
            card = 'yellow'
            reasoning.append("Conduta imprudente - Cartão Amarelo")
        elif severity_score > 0.3:
            card = 'foul_only'
            reasoning.append("Falta descuidada - Apenas falta")
        
        # Considerações especiais para área penal
        if is_penalty_area and severity_score > 0.3:
            reasoning.append("Falta cometida na área penal - PÊNALTI")
        
        return {
            'severity_score': severity_score,
            'card': card,
            'is_penalty': is_penalty_area and severity_score > 0.3,
            'reasoning': reasoning,
            'confidence': foul_classification['confidence']
        }

if __name__ == "__main__":
    # Teste básico
    classifier = FoulClassifier()
    print(f"Classificador de faltas inicializado com sucesso!")
    print(f"Dispositivo: {classifier.device}")
    print(f"Modelo: {classifier.model_type}")

