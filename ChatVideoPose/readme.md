
# ChatVideoPose

## Pose-LLM Integration

This repository implements a modular architecture for integrating pose estimation with large language models (LLMs) to answer pose-related queries.

## Architecture

The system processes video frames through the following pipeline:
1. **Pose Estimation**: Extract pose features from video frames
2. **Positional Embedding**: Embed pose features for better representation
3. **Q-Former**: Transform pose embeddings
4. **Linear Layer**: Final transformation of pose information
5. **LLM Integration**: Process user queries along with pose information via an LLM
6. **SMPL Token**: Special tokens in the LLM space representing pose information

## Installation

```bash
git clone https://github.com/Saibhargav1208/pose-llm.git
cd pose-llm
pip install -r requirements.txt
```

## Usage

```bash
python main.py --config config/config.yaml --video path/to/video.mp4 --query "Describe the movement in this video"
```

## Configuration

Edit `config/config.yaml` to customize:
- Pose estimation model
- LLM model choice
- Q-Former parameters
- Other system settings

## Requirements

See `requirements.txt` for all dependencies. Major requirements include:
- PyTorch
- OpenCV
- Transformers
- (Your chosen pose estimation library)

## Structure

The codebase is organized into modular components:
- `src/pose/`: Pose estimation modules
- `src/embedding/`: Positional embedding implementation
- `src/models/`: Q-Former and Linear Layer implementations
- `src/llm/`: LLM integration and space handling
- `src/query/`: Query processing
- `src/utils/`: Helper functions and visualization tools

## License

[Your chosen license]
