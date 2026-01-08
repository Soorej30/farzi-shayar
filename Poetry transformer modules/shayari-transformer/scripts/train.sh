poetry-generator/
│
├── data/
│   ├── raw/                     # Original raw data files
│   │   └── shayari_db.csv      # The original CSV file
│   ├── processed/               # Processed data files
│   │   └── cleaned_shayari_db.csv  # Cleaned and preprocessed data
│   └── tokenizer/               # Tokenizer files (if applicable)
│
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── data_exploration.ipynb   # Notebook for exploring the data
│   └── model_training.ipynb      # Notebook for training the model
│
├── src/                         # Source code for the project
│   ├── __init__.py              # Makes src a package
│   ├── data_preprocessing.py     # Script for data cleaning and preprocessing
│   ├── model.py                  # Model architecture (transformer)
│   ├── train.py                  # Training script
│   ├── generate.py               # Script for generating poetry
│   ├── utils.py                  # Utility functions (e.g., for loading data, saving models)
│   └── config.py                 # Configuration file for hyperparameters and settings
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Files to ignore in git