poetry-generator/
│
├── data/
│   ├── raw/                     # Raw data files (e.g., shayari_db.csv)
│   ├── processed/               # Processed data files (e.g., tokenized data)
│   └── README.md                # Description of the data structure
│
├── notebooks/                   # Jupyter notebooks for exploration and experimentation
│   ├── data_exploration.ipynb   # Notebook for exploring the dataset
│   ├── model_training.ipynb      # Notebook for training the model
│   └── poetry_generation.ipynb   # Notebook for generating poetry
│
├── src/                         # Source code for the project
│   ├── __init__.py              # Makes src a package
│   ├── data_preprocessing.py     # Data preprocessing functions
│   ├── model.py                  # Model architecture (e.g., Transformer)
│   ├── train.py                  # Training script
│   ├── generate.py               # Poetry generation script
│   ├── utils.py                  # Utility functions (e.g., for loading data, saving models)
│   └── config.py                 # Configuration settings (hyperparameters, paths, etc.)
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview and instructions
└── .gitignore                    # Files and directories to ignore in version control