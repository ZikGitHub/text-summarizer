from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from textSummarizer.entity import ModelTrainerConfig
import os
from datasets import load_dataset, load_from_disk
import torch


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt
        ).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        # loading data
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=500,
            save_steps=1e6,
            gradient_accumulation_steps=16,
            report_to=[],  # This disables MLflow, WandB, and other integrations
        )

        trainer = Trainer(
            data_collator=seq2seq_data_collator,
            model=model_pegasus,
            args=trainer_args,
            tokenizer=tokenizer,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"],
            callbacks=[],
        )

        trainer.train()

        ## Save model
        model_pegasus.save_pretrained(
            os.path.join(self.config.root_dir, "pegasus-samsum-model")
        )
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
