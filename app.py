"""
Web Interface for the AI Football Referee Analysis System
"""
import gradio as gr
from pathlib import Path
import tempfile

from video_analyzer import FootballVideoAnalyzer

class FootballRefereeApp:
    """Web application for analyzing football videos"""
    
    def __init__(self):
        """Initializes the application"""
        self.analyzer = None
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def initialize_analyzer(self):
        """Initializes the analyzer (lazy loading)"""
        if self.analyzer is None:
            print("Initializing analysis system...")
            self.analyzer = FootballVideoAnalyzer()
        return self.analyzer
    
    def analyze_video_interface(self, video_file, sample_rate):
        """
        Interface for video analysis
        
        Args:
            video_file: Uploaded video file
            sample_rate: Frame sampling rate
            
        Returns:
            Tuple with results
        """
        if video_file is None:
            return "⚠️ Please upload a video.", "", "", None
        
        try:
            analyzer = self.initialize_analyzer()
            results_dir = self.temp_dir / "results"
            results_dir.mkdir(exist_ok=True)
            
            results = analyzer.analyze_video(
                video_path=video_file,
                output_dir=str(results_dir),
                sample_rate=int(sample_rate)
            )
            
            summary = self._generate_summary(results)
            
            explanation_path = results_dir / 'decision_explanation.txt'
            with open(explanation_path, 'r', encoding='utf-8') as f:
                explanation = f.read()
            
            annotated_video = self._create_annotated_video(video_file, results)
            
            json_path = results_dir / 'analysis_results.json'
            with open(json_path, 'r', encoding='utf-8') as f:
                json_results = f.read()
            
            return summary, explanation, json_results, annotated_video
            
        except Exception as e:
            error_msg = f"❌ Error analyzing video: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, "", "", None
    
    def _generate_summary(self, results: dict) -> str:
        """
        Generates a visual summary of the analysis
        """
        decision = results["decision"]
        analysis = results["analysis"]
        video_info = results["video_info"]
        
        decision_emoji = {
            'no_foul': '✅',
            'foul': '⚠️',
            'penalty': '🎯',
            'yellow_card': '🟨',
            'red_card': '🟥'
        }
        
        card_emoji = {
            'none': '',
            'yellow': '🟨',
            'red': '🟥'
        }
        
        summary = ["# 🏆 Analysis Result"]
        summary.append("\n---\n")
        
        emoji = decision_emoji.get(decision.decision.value, '❓')
        summary.append(f"## {emoji} DECISION: **{decision.decision.value.upper().replace('_', ' ')}**")
        
        confidence_bar = "█" * int(decision.confidence * 10) + "░" * (10 - int(decision.confidence * 10))
        summary.append(f"**Confidence:** {confidence_bar} {decision.confidence:.1%}")
        
        if decision.card.value != 'none':
            card_icon = card_emoji[decision.card.value]
            summary.append(f"### {card_icon} CARD: {decision.card.value.upper()}")
        
        if decision.is_penalty:
            summary.append("### ⚽ PENALTY KICK")
        
        summary.append("\n---\n")
        summary.append("## 📊 Analysis Statistics")
        summary.append(f"- **Video Duration:** {video_info['duration']:.2f}s")
        summary.append(f"- **Analyzed Frames:** {video_info['analyzed_frames']} of {video_info['total_frames']}")
        summary.append(f"- **Interaction Frames:** {analysis['interaction_frames']}")
        summary.append(f"- **Predominant Action:** {analysis['predominant_action']}")
        summary.append(f"- **Contact Intensity:** {analysis['contact_intensity']:.1%}")
        summary.append(f"- **Classification:** {analysis['foul_classification']['class']}")
        
        summary.append("\n---\n")
        summary.append("## 🔍 Key Reasons")
        for i, reason in enumerate(decision.reasoning[:5], 1):
            summary.append(f"{i}. {reason}")
        
        if decision.rule_references:
            summary.append("\n## 📖 Rules Applied")
            for rule in decision.rule_references:
                summary.append(f"- {rule}")
        
        return "\n".join(summary)
    
    def _create_annotated_video(self, video_path: str, results: dict) -> str:
        """Creates an annotated version of the video"""
        return video_path
    
    def create_interface(self):
        """
        Creates the Gradio interface
        """
        theme = gr.themes.Soft(primary_hue="blue", secondary_hue="green")
        
        with gr.Blocks(theme=theme, title="AI Football Referee") as interface:
            gr.Markdown("""
            # ⚽ AI Football Referee - Play Analysis Assistant
            
            A computer vision system to assist referees in analyzing football plays.
            The AI analyzes videos to determine:
            - ✅ If a foul occurred
            - 🟨 If a yellow card is warranted
            - 🟥 If a red card is warranted
            - 🎯 If it is a penalty
            
            **Based on the official IFAB (International Football Association Board) Laws of the Game.**
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Upload Video")
                    video_input = gr.Video(label="Football Play Video", sources=["upload"], height=400)
                    sample_rate = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Sample Rate (analyze 1 of N frames)", info="Higher values = faster, less precise analysis")
                    analyze_btn = gr.Button("🔍 Analyze Play", variant="primary", size="lg")
                    gr.Markdown("""
                    ---
                    ### 💡 Tips:
                    - Short clips (5-15s) work best.
                    - Ensure players are clearly visible.
                    - The clip should show the moment of contact.
                    """)
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📋 Analysis Results")
                    summary_output = gr.Markdown(label="Summary", value="Awaiting analysis...")
                    with gr.Accordion("📄 Detailed Explanation", open=False):
                        explanation_output = gr.Textbox(label="Full Explanation", lines=15, max_lines=30)
                    with gr.Accordion("🎥 Annotated Video", open=False):
                        video_output = gr.Video(label="Video with Annotations")
                    with gr.Accordion("💾 JSON Data", open=False):
                        json_output = gr.Code(label="JSON Results", language="json")
            
            gr.Markdown("---")
            gr.Markdown("""
            ### 📚 About the System
            
            This system uses multiple AI technologies:
            
            1. **YOLOv8**: For real-time player and ball detection.
            2. **MediaPipe**: For player pose and movement analysis.
            3. **CNN (MobileNetV2)**: For foul classification.
            4. **Rules Engine**: To apply official IFAB rules.
            
            **Based on:**
            - Official IFAB Laws of the Game 2025/2026
            - Scientific research in automated foul detection
            - State-of-the-art deep learning models
            """)
            
            analyze_btn.click(
                fn=self.analyze_video_interface,
                inputs=[video_input, sample_rate],
                outputs=[summary_output, explanation_output, json_output, video_output]
            )
        
        return interface
    
    def launch(self, share=False, server_port=7860):
        """
        Launches the application
        """
        interface = self.create_interface()
        interface.launch(share=share, server_name="0.0.0.0", server_port=server_port, show_error=True)

def main():
    """Main function"""
    print("="*60)
    print("AI Football Referee")
    print("Football Play Analysis System")
    print("="*60)
    print()
    
    app = FootballRefereeApp()
    app.launch(share=True, server_port=7860)

if __name__ == "__main__":
    main()

