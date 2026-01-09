# farzi-shayar
Transformer based model to create shayaris/ghazals

To train the shayari model

We open a CLI in the **shayari-transformer** folder, and run
- python -m src.run --mode train --epochs <num_epochs> --batch_size <batch_size> --max_len <max_length_of_poetry>

As a generic example, you can try
- python -m src.run --mode train --epochs 10 --batch_size 32 --max_len 64

Once the training is completed, to generate a poetry, open a CLI in the **shayari-transformer** folder, and run
- python -m src.run --mode generate --model_path <trained_model_path> --start_word "<start_word>" --generate_max_len <length>

As a generic example, you can try
- python -m src.run --mode generate --model_path models/transformer_poetry.keras --start_word "tera" --generate_max_len 40