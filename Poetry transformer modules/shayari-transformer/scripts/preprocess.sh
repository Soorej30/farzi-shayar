poetry-generator/
│
├── data/
│   ├── raw/                     # Raw data files
│   │   └── shayari_db.csv       # Original poetry data
│   ├── processed/               # Processed data files
│   │   └── cleaned_shayari_db.csv # Cleaned and preprocessed poetry data
│   └── tokenizer/               # Tokenizer files (if applicable)
│       └── tokenizer.json        # Tokenizer configuration
│
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── data_exploration.ipynb   # Data exploration and visualization
│   └── model_training.ipynb      # Model training experiments
│
├── src/                         # Source code for the project
│   ├── __init__.py              # Makes src a package
│   ├── data/                    # Data handling scripts
│   │   ├── data_loader.py        # Load and preprocess data
│   │   └── data_preprocessing.py  # Functions for cleaning and preprocessing
│   ├── model/                   # Model-related scripts
│   │   ├── transformer_model.py   # Transformer model definition
│   │   ├── trainer.py            # Training loop and evaluation
│   │   └── generator.py          # Poetry generation logic
│   ├── utils/                   # Utility functions
│   │   ├── config.py             # Configuration settings
│   │   └── logger.py             # Logging utilities
│   └── main.py                   # Main entry point for training and generation
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Files to ignore in version control