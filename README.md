# RoadRep - Road Representation Learning

RoadRep is a project for learning road representations using deep learning models, specifically focusing on CLIP-based architectures. This repository contains the code for model inference, training, and deployment.

## Project Structure

```
RoadRep/
├── models/                  # ONNX model files
├── src/                     # Source code
│   ├── __init__.py         # Package initialization
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── model.py            # Model loading and inference
│   ├── inference.py        # Inference pipeline
│   ├── train.py            # Model training
│   └── utils.py            # Utility functions
├── configs/                # Configuration files
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RoadRep
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Inference

To run inference on an image:

```bash
python -m src.inference --image_path path/to/your/image.jpg --model_dir models/
```

Optional arguments:
- `--quantized`: Use the quantized model (smaller, faster, slightly less accurate)
- `--output_path`: Path to save the visualization

### Training

1. Prepare your dataset and create a configuration file (see `configs/example_config.yaml`).

2. Run training:
   ```bash
   python -m src.train --config configs/your_config.yaml --experiment_name my_experiment
   ```

### Docker

To build and run the Docker container:

```bash
# Build the Docker image
docker build -t roadrep .

# Run the container
docker run --gpus all -p 8000:8000 roadrep
```

## Model Details

The project uses a CLIP-based model for road representation learning. The model is provided in two versions:
- `roadrep_clip.onnx`: Full-precision model (larger, more accurate)
- `roadrep_clip_int8.onnx`: Quantized model (smaller, faster, slightly less accurate)

## License

[Your License Here]

## Citation

If you use this project in your research, please cite:

```
[Your Citation Here]
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

[Your Contact Information]
