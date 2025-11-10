# ⚽ AI Football Referee

![AI Referee](https://i.imgur.com/8a13g1h.png)

**AI Football Referee** is a computer vision system designed to assist football referees in analyzing plays. It uses deep learning to detect fouls, determine yellow/red cards, and identify penalties based on the official IFAB Laws of the Game.

This project leverages state-of-the-art AI models to provide a real-time, data-driven second opinion for critical in-game incidents.

## ✨ Features

- ✅ **Foul Detection**: Identifies illegal contact and determines if a foul occurred.
- 🟨 **Yellow Card Analysis**: Recommends a yellow card for reckless challenges and unsporting behavior.
- 🟥 **Red Card Analysis**: Recommends a red card for serious foul play and excessive force.
- 🎯 **Penalty Detection**: Automatically identifies fouls committed by the defending team inside the penalty area.
- 🎥 **Video Analysis**: Processes video clips of football plays to generate a decision.
- 📊 **Detailed Explanations**: Provides a clear rationale for each decision, referencing specific IFAB rules.
- 🧤 **Goalkeeper Intelligence**: Automatically fetches goalkeeper stats (height, save%) via API or web scraping to add context to the analysis.

## 🚀 Live Demo

The application is running and can be accessed through the Gradio interface. You can upload your own video clips to test the AI's decisions.

*(Note: The live demo link is generated when running `app.py`)*

## 🛠️ Tech Stack & Architecture

The system is built with a modular architecture, where each component is responsible for a specific task in the analysis pipeline.

```
┌─────────────────────────────────────────────────────────┐
│                    Interface (Gradio)                   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              VideoAnalyzer (Orchestrator)               │
└─────────┬────────┬──────────┬──────────┬────────────────┘
          │        │          │          │            
          ▼        ▼          ▼          ▼           
      ┌──────┐ ┌──────┐ ┌────────┐ ┌────────┐ 
      │YOLOv8│ │Media-│ │Mobile- │ │Rules   │ 
      │      │ │Pipe  │ │NetV2   │ │Engine  │ 
      └──────┘ └──────┘ └────────┘ └────────┘ 
      Detection  Pose    Classif.   IFAB Rules  
```

### Core Components

1.  **Player Detector** (`player_detector.py`): Uses **YOLOv8** to perform real-time object detection, identifying players and the ball in each frame.
2.  **Pose Analyzer** (`pose_analyzer.py`): Employs **MediaPipe** to analyze player poses and movements, determining the type of action (e.g., tackle, push, kick) and the intensity of physical contact.
3.  **Foul Classifier** (`foul_classifier.py`): A Convolutional Neural Network (**MobileNetV2**) trained to classify frames into *no foul*, *minor foul*, or *serious foul* based on the visual evidence.
4.  **Rules Engine** (`rules_engine.py`): A logic module that implements the official **IFAB Laws of the Game**, specifically Law 12 (Fouls and Misconduct) and Law 14 (The Penalty Kick). It uses the output from the other modules to make a final, rule-based decision.
5.  **Video Analyzer** (`video_analyzer.py`): The central orchestrator that manages the entire analysis pipeline, from video processing to generating the final verdict.

### Technologies Used

- **Deep Learning**: PyTorch, YOLOv8, MediaPipe, MobileNetV2
- **Computer Vision**: OpenCV
- **Web Interface**: Gradio
- **Data Handling**: NumPy, Requests, BeautifulSoup
- **Language**: Python 3.11

## ⚙️ Setup and Installation

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/football-referee-ai.git
cd football-referee-ai
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

The application will start and provide a local URL (usually `http://127.0.0.1:7860`) to access the web interface.

## 📖 How to Use

1.  **Launch the app** and open the provided URL in your browser.
2.  **Upload a video clip** of a football play.
3.  **Adjust the sample rate** (a lower value is more accurate but slower).
4.  Click **"Analyze Play"** and wait for the processing to complete.
5.  **Review the results**, which include the final decision, confidence level, and a detailed explanation.

## 📚 Project Structure

```
/football-referee-ai
├── app.py                      # Gradio web interface
├── video_analyzer.py           # Main analysis orchestrator
├── player_detector.py          # YOLOv8 object detection
├── pose_analyzer.py            # MediaPipe pose analysis
├── foul_classifier.py          # CNN-based foul classification
├── rules_engine.py             # IFAB rules logic
├── requirements.txt            # Project dependencies
├── models/                     # (Directory for trained models)
├── data/                       # (Directory for data and cache)
├── uploads/                    # (Directory for uploaded videos)
└── results/                    # (Directory for analysis results)
```

## 🎓 Scientific Foundation

This project is grounded in official football regulations and academic research in sports analytics.

- **IFAB Laws of the Game**: The core logic is based on the official rules published by The International Football Association Board. [1]
- **Deep Learning for Foul Detection**: The classification model architecture is inspired by research demonstrating the effectiveness of CNNs in automatically detecting fouls in sports, achieving high accuracy with models like MobileNetV2. [2]

## ⚠️ Limitations

- **Dataset**: The models are pre-trained on general datasets. Fine-tuning on a large, specific dataset of football fouls would significantly improve accuracy.
- **Subjectivity**: Football refereeing has an inherent subjective element that cannot be fully captured by an algorithm.
- **Video Quality**: The system's performance is highly dependent on the quality and angle of the input video.

## 🔮 Future Work

- [ ] **Fine-tune Models**: Train the detection and classification models on a custom dataset of football plays.
- [ ] **Player Tracking**: Implement temporal tracking to follow players across frames.
- [ ] **Simulation Detection**: Develop a model to identify player simulations ("diving").
- [ ] **UI/UX Enhancements**: Add features like drawing on the video and comparing plays side-by-side.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### References

[1] The International Football Association Board. *Laws of the Game*. [https://www.theifab.com/laws/](https://www.theifab.com/laws/)

[2] Rabee, A., et al. (2025). *Comparative analysis of automated foul detection in football using deep learning architectures*. Scientific Reports.

