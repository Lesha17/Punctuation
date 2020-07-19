# Demo

Evaluation and results are shown in Demo.ipynb notebook. To to re-run this notebook, put model.zip in the working directory 
and ensure that there is an internet connection from notebook runtime.

# Model weights

Trained model weights are available from https://drive.google.com/file/d/1uenebpycXgj82WPGUUsL-SorvGg4YT8n/view?usp=sharing

# Training

Model is trained with script

```bash
train_transformers.sh --data_dir=data/news/Lenta/ --output_dir=./models/Lenta_10epochs --num_train_epochs=10 --learning_rate=3e-5
```

Arguments:

**--data_dir** - directory with train/, test and dev/ subdirectories, each must contain raw .txt files to train or eval model on

All other arguments are passed into [transformers.TrainingArguments](https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py#L33)
