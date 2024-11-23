# IntelliDoc

A deep learning project implementing advanced document classification using Large Language Models (LLMs). The system leverages multi-layer supervision and attention mechanisms to achieve high-accuracy document categorization, featuring comprehensive performance monitoring and visualization capabilities.

## Features

- Advanced LLM-based document classification
- Multi-layer supervision with weighted loss aggregation
- Attention pattern visualization and analysis
- Comprehensive performance tracking across model layers
- Robust data preprocessing pipeline
- Extensive evaluation metrics and reporting

## Installation

This project uses Poetry for dependency management. To get started:

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd intellidoc
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

## Project Structure

```
intellidoc/
├── data/                    # Data storage directory
├── intellidoc/             # Main package directory
│   ├── data_processing/    # Data preprocessing modules
│   ├── models/             # Model architectures
│   ├── training/           # Training modules
│   ├── evaluation/         # Evaluation utilities
│   └── utilities/          # Helper functions
├── tests/                  # Test suite
└── poetry.lock            # Dependency lock file
```

## Usage

1. Data Preparation:
   ```python
   from intellidoc.data_processing import data_preprocessing
   # Prepare your data using the preprocessing pipeline
   ```

2. Training:
   ```python
   from intellidoc.training.llm_trainer import train
   # Execute the training pipeline
   train()
   ```

3. Evaluation:
   ```python
   # Basic evaluation
   from intellidoc.evaluation.roberta_evaluation import evaluate_model
   evaluate_model()

   # Advanced LLM evaluation with detailed metrics
   from intellidoc.evaluation.advance_llm_evaluation import eval
   eval()
   ```

The advanced LLM evaluation provides:
- Detailed performance metrics with label-wise analysis
- Confidence score distribution
- Analysis of incorrect predictions
- JSON export of evaluation results

## Requirements

- Python 3.11
- PyTorch with torchtext
- Transformers
- scikit-learn
- pandas
- tqdm
- Other dependencies as specified in pyproject.toml

## Development

1. Run tests:
   ```bash
   poetry run pytest
   ```

2. Check code style:
   ```bash
   poetry run black .
   poetry run isort .
   ```

## License

This project is licensed under (MIT License) the terms of the LICENSE file included in the repository.

## Author

Moitree Basu (sfurti.basu@gmail.com)