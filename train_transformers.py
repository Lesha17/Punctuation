import argparse
import os

import transformers
from transformers.trainer import get_tpu_sampler
from transformers import TokenClassificationPipeline

from data_reader import PunctuationDataset
from utils import PUNCT_TO_ID, calc_metrics

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, IterableDataset


class CustomTrainer(transformers.Trainer):
    def get_train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_dataset, self.args.train_batch_size)

    def get_eval_dataloader(self, dataset) -> DataLoader:
        return self.get_dataloader(dataset if dataset is not None else self.eval_dataset, self.args.eval_batch_size)

    def get_dataloader(self, dataset, batch_size):
        if dataset is None:
            raise ValueError("Trainer: requires a dataset.")
        if transformers.is_torch_tpu_available():
            sampler = get_tpu_sampler(dataset)
        elif not isinstance(dataset, IterableDataset):
            sampler = (
                RandomSampler(dataset)
                if self.args.local_rank == -1
                else DistributedSampler(dataset)
            )
        else:
            sampler = None

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
        )

        return data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    default_training_args = vars(transformers.TrainingArguments(
        output_dir="./models/rubert_cased_nplus1",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        num_train_epochs=5,
        learning_rate=2e-5,
        logging_steps=500,
        logging_first_step=True,
        save_steps=1000,
        evaluate_during_training=True,
    ))
    for k, v in default_training_args.items():
        parser.add_argument('--' + k, default=v, type=type(v))
    args = parser.parse_args()
    training_args_dict = {k: v for k, v in vars(args).items() if k in default_training_args}

    data_dir = args.data_dir

    tokenizer = transformers.AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    config = transformers.AutoConfig.from_pretrained('DeepPavlov/rubert-base-cased')
    config.num_labels = len(PUNCT_TO_ID)
    model = transformers.AutoModelForTokenClassification.from_pretrained('DeepPavlov/rubert-base-cased', config=config)

    dataset_train = PunctuationDataset(data_dir=os.path.join(data_dir, 'train'), tokenizer=tokenizer,
                                       label_to_idx=PUNCT_TO_ID, batch_size=training_args_dict['per_device_train_batch_size'])
    dataset_test = PunctuationDataset(data_dir=os.path.join(data_dir, 'test'), tokenizer=tokenizer,
                                      label_to_idx=PUNCT_TO_ID, batch_size=training_args_dict['per_device_eval_batch_size'])
    dataset_dev = PunctuationDataset(data_dir=os.path.join(data_dir, 'dev'), tokenizer=tokenizer,
                                     label_to_idx=PUNCT_TO_ID, batch_size=training_args_dict['per_device_eval_batch_size'])

    training_args = transformers.TrainingArguments(**training_args_dict)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        compute_metrics=calc_metrics,
        data_collator=transformers.default_data_collator
    )

    trainer.train()

    trainer.save_model()