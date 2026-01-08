poetry-generator/
│
├── data/
│   ├── raw/                     # Raw data files (e.g., shayari_db.csv)
│   ├── processed/               # Processed data files (e.g., tokenized data)
│   └── README.md                # Description of the data
│
├── notebooks/                   # Jupyter notebooks for exploration and experimentation
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── poetry_generation.ipynb
│
├── src/                         # Source code for the project
│   ├── __init__.py              # Makes src a package
│   ├── data_preprocessing.py     # Data cleaning and preprocessing functions
│   ├── model.py                  # Model definition (transformer architecture)
│   ├── train.py                  # Training script
│   ├── generate.py               # Poetry generation script
│   ├── utils.py                  # Utility functions (e.g., for tokenization)
│   └── config.py                 # Configuration settings (hyperparameters, paths)
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview and instructions
└── main.py                       # Main entry point for training and generation