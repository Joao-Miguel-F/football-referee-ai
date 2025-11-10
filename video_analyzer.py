"""
Sistema Integrado de Análise de Vídeo de Futebol
Combina detecção, pose, classificação e regras
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from player_detector import PlayerDetector
from pose_analyzer import PoseAnalyzer
from foul_classifier import FoulClassifier, FoulSeverityAnalyzer
from rules_engine import FootballRulesEngine, IncidentContext, RefereeDecision

class FootballVideoAnalyzer:
    """Analisador completo de vídeos de futebol"""
    
    def __init__(self):
        """Inicializa todos os componentes do sistema"""
        print("Inicializando sistema de análise de vídeo...")
        
        self.player_detector = PlayerDetector()
        self.pose_analyzer = PoseAnalyzer()
        self.foul_classifier = FoulClassifier()
        self.severity_analyzer = FoulSeverityAnalyzer()
        self.rules_engine = FootballRulesEngine()
        
        print("Sistema inicializado com sucesso!")
    
    def analyze_video(self, video_path: str, 
                     output_dir: str = "results",
                     sample_rate: int = 3) -> Dict:
        """
        Analisa vídeo completo de lance de futebol
        
        Args:
            video_path: Caminho para o vídeo
            output_dir: Diretório para salvar resultados
            sample_rate: Analisar 1 a cada N frames
            
        Returns:
            Dict com análise completa
        """
        print(f"\n{'='*60}")
        print(f"Analisando vídeo: {video_path}")
        print(f"{'='*60}\n")
        
        # Abrir vídeo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
        
        # Informações do vídeo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        print(f"Informações do vídeo:")
        print(f"  - Resolução: {width}x{height}")
        print(f"  - FPS: {fps}")
        print(f"  - Duração: {duration:.2f}s ({total_frames} frames)")
        print(f"  - Taxa de amostragem: 1/{sample_rate} frames\n")
        
        # Análise frame por frame
        frame_analyses = []
        frames_for_classification = []
        frame_count = 0
        analyzed_count = 0
        
        print("Processando frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Amostrar frames
            if frame_count % sample_rate != 0:
                continue
            
            analyzed_count += 1
            
            # Análise do frame
            frame_analysis = self._analyze_frame(frame, frame_count)
            frame_analyses.append(frame_analysis)
            
            # Guardar frames com interação detectada
            if frame_analysis['has_interaction']:
                frames_for_classification.append(frame)
            
            # Progresso
            if analyzed_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  Progresso: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        cap.release()
        
        print(f"\nFrames analisados: {analyzed_count}")
        print(f"Interações detectadas: {len(frames_for_classification)}\n")
        
        # Análise agregada
        print("Realizando análise agregada...")
        final_analysis = self._aggregate_analysis(frame_analyses, frames_for_classification)
        
        # Gerar decisão final
        print("Gerando decisão final do árbitro...\n")
        decision = self._generate_decision(final_analysis)
        
        # Salvar resultados
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        self._save_results(final_analysis, decision, output_path)
        
        return {
            'video_info': {
                'path': video_path,
                'duration': duration,
                'total_frames': total_frames,
                'analyzed_frames': analyzed_count
            },
            'analysis': final_analysis,
            'decision': decision,
            'output_dir': str(output_path)
        }
    
    def _analyze_frame(self, frame: np.ndarray, frame_number: int) -> Dict:
        """
        Analisa um frame individual
        
        Args:
            frame: Frame do vídeo
            frame_number: Número do frame
            
        Returns:
            Dict com análise do frame
        """
        # 1. Detectar jogadores
        detections = self.player_detector.detect_frame(frame)
        players = detections['players']
        ball = detections['ball']
        
        # 2. Verificar interação entre jogadores
        close_groups = self.player_detector.detect_close_players(players)
        has_interaction = len(close_groups) > 0
        
        analysis = {
            'frame_number': frame_number,
            'players_detected': len(players),
            'ball_detected': len(ball) > 0,
            'has_interaction': has_interaction,
            'close_groups': close_groups,
            'detections': detections
        }
        
        # 3. Se há interação, analisar mais profundamente
        if has_interaction and len(players) >= 2:
            # Extrair região de interação
            interaction_region = self.player_detector.get_interaction_region(frame, players)
            
            # Analisar poses
            poses = []
            for i, (x1, y1, x2, y2, conf) in enumerate(players[:4]):  # Limitar a 4 jogadores
                player_crop = frame[y1:y2, x1:x2]
                if player_crop.size > 0:
                    pose_data = self.pose_analyzer.analyze_pose(player_crop)
                    if pose_data:
                        action_type = self.pose_analyzer.detect_action_type(pose_data)
                        poses.append({
                            'player_id': i,
                            'action_type': action_type,
                            'pose_data': pose_data
                        })
            
            analysis['poses'] = poses
            analysis['interaction_region'] = interaction_region
        
        return analysis
    
    def _aggregate_analysis(self, frame_analyses: List[Dict], 
                           interaction_frames: List[np.ndarray]) -> Dict:
        """
        Agrega análises de todos os frames
        
        Args:
            frame_analyses: Lista de análises de frames
            interaction_frames: Frames com interação detectada
            
        Returns:
            Análise agregada
        """
        # Estatísticas gerais
        total_players = sum(f['players_detected'] for f in frame_analyses)
        avg_players = total_players / len(frame_analyses) if frame_analyses else 0
        
        interaction_frames_count = sum(1 for f in frame_analyses if f['has_interaction'])
        
        # Detectar ações predominantes
        action_types = []
        for frame in frame_analyses:
            if 'poses' in frame:
                for pose in frame['poses']:
                    action_types.append(pose['action_type'])
        
        # Classificar sequência de frames com interação
        foul_classification = {'class': 'no_foul', 'confidence': 0.0}
        if interaction_frames:
            print(f"  Classificando {len(interaction_frames)} frames com interação...")
            foul_classification = self.foul_classifier.classify_sequence(interaction_frames)
        
        # Calcular intensidade média de contato
        contact_intensity = 0.0
        if interaction_frames_count > 0:
            # Estimativa baseada na proporção de frames com interação
            contact_intensity = min(1.0, interaction_frames_count / len(frame_analyses) * 2)
        
        return {
            'total_frames_analyzed': len(frame_analyses),
            'avg_players_per_frame': avg_players,
            'interaction_frames': interaction_frames_count,
            'action_types': action_types,
            'predominant_action': max(set(action_types), key=action_types.count) if action_types else 'unknown',
            'foul_classification': foul_classification,
            'contact_intensity': contact_intensity
        }
    
    def _generate_decision(self, analysis: Dict) -> RefereeDecision:
        """
        Gera decisão final do árbitro baseada na análise
        
        Args:
            analysis: Análise agregada
            
        Returns:
            Decisão do árbitro
        """
        # Criar contexto do incidente
        context = IncidentContext(
            location='midfield',  # Simplificação - em produção, detectar área do campo
            action_type=analysis['predominant_action'],
            contact_intensity=analysis['contact_intensity'],
            ball_proximity=0.5,  # Simplificação
            player_movement='towards_ball',
            body_position='upright',
            has_possession=False,
            denies_goal_opportunity=False
        )
        
        # Gerar decisão usando motor de regras
        decision = self.rules_engine.analyze_incident(
            context=context,
            foul_classification=analysis['foul_classification']
        )
        
        return decision
    
    def _save_results(self, analysis: Dict, decision: RefereeDecision, 
                     output_dir: Path):
        """
        Salva resultados da análise
        
        Args:
            analysis: Análise agregada
            decision: Decisão do árbitro
            output_dir: Diretório de saída
        """
        # Salvar JSON
        results = {
            'analysis': {
                'total_frames': analysis['total_frames_analyzed'],
                'interaction_frames': analysis['interaction_frames'],
                'predominant_action': analysis['predominant_action'],
                'contact_intensity': analysis['contact_intensity'],
                'foul_classification': analysis['foul_classification']
            },
            'decision': {
                'verdict': decision.decision.value,
                'card': decision.card.value,
                'is_penalty': decision.is_penalty,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'rule_references': decision.rule_references
            }
        }
        
        json_path = output_dir / 'analysis_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Salvar explicação textual
        explanation = self.rules_engine.explain_decision(decision)
        txt_path = output_dir / 'decision_explanation.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(explanation)
        
        print(f"Resultados salvos em: {output_dir}")
        print(f"  - {json_path.name}")
        print(f"  - {txt_path.name}")

if __name__ == "__main__":
    # Teste básico
    analyzer = FootballVideoAnalyzer()
    print("\nSistema pronto para análise de vídeos!")

