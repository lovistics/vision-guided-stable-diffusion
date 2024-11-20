# Vision-Guided Stable Diffusion with ControlNet

A computer vision pipeline that leverages facial landmark detection and pose estimation to guide Stable Diffusion image generation through ControlNet conditioning. The system performs multi-feature analysis using dlib and MediaPipe to control the image generation process.

## Features

- 🎯 Facial landmark detection with dlib (68 points)
- 🏃‍♂️ Full-body pose estimation via MediaPipe
- 🖼️ Controlled image generation using Stable Diffusion + ControlNet
- 🔄 Automatic pose-guided conditioning
- ✨ CLAHE-enhanced image processing
- 🎨 Multiple generation styles (Realistic, Artistic, Cartoon, Fantasy)

## Installation

```bash
# Clone the repository
git clone https://github.com/lovistics/vision-guided-stable-diffusion.git
cd vision-guided-stable-diffusion

# Install dependencies
pip install -r requirements.txt

# Download required model
python setup.py
```

## Project Structure
```
vision-guided-stable-diffusion/
├── models/
│   └── shape_predictor_68_face_landmarks.dat
├── analyzer.py          # Facial landmark detection
├── generator.py         # Stable Diffusion + ControlNet
├── gpt.py              # GPT-4 integration
├── interface.py        # Gradio UI
├── main.py             # Core pipeline
├── pose.py             # Pose detection
├── processor.py        # Image preprocessing
├── run.py             # Application entry point
├── setup.py           # Model setup
└── requirements.txt
```

## Usage

1. Start the application:
```bash
python run.py
```

2. Access the web interface through your browser
3. Upload an image
4. Wait for the analysis and generation process
5. View the results in the gallery

## Technical Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- OpenAI API key
- 8GB+ GPU RAM recommended

## Core Components

### Vision Analysis
- Face landmark detection (dlib)
- Pose estimation (MediaPipe)
- Feature extraction pipeline

### Generation Pipeline
- Stable Diffusion v1.5
- ControlNet conditioning
- Automatic post-processing

### Processing Pipeline
- CLAHE enhancement
- Aspect ratio preservation
- Automatic pose map generation

## Model Configuration

Key parameters:
- Face Detection Confidence: 0.5
- Pose Detection Threshold: 0.5
- ControlNet Conditioning Scale: 0.8
- Generation Steps: 30-50

## Performance Notes

- GPU acceleration recommended for real-time performance
- Average processing time: 3-5s per image (with GPU)
- Memory optimization with torch.cuda.amp and xformers

## Environment Variables

Required environment variables:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Known Issues & Troubleshooting

- If you see CUDA out of memory errors, try reducing the image size in config
- For "accelerate not found" warning, install with: `pip install accelerate`
- Windows users may need to run Python as administrator for symlink support

## License

MIT License. See LICENSE for details.

## Citation

```bibtex
@software{vision_guided_sd,
  title = {Vision-Guided Stable Diffusion with ControlNet},
  author = {Lovice Ochieng},
  year = {2024},
  url = {https://github.com/lovistics/vision-guided-stable-diffusion.git}
}
```

## Acknowledgments

- [dlib](http://dlib.net/) for facial landmark detection
- [MediaPipe](https://mediapipe.dev/) for pose estimation
- [Stable Diffusion](https://stability.ai/) for image generation
- [ControlNet](https://github.com/lllyasviel/ControlNet) for guided generation
