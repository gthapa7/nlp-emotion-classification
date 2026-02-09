from pathlib import Path
import re

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	DataCollatorWithPadding,
	TrainingArguments,
	Trainer,
)

from src.evaluation import save_metrics_row


def load_split(file_path: Path) -> pd.DataFrame:
	return pd.read_csv(
		file_path,
		sep=";",
		header=None,
		names=["text", "emotion"],
	)


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
	data_dir = Path(data_dir)
	train_df = load_split(data_dir / "train.txt")
	test_df = load_split(data_dir / "test.txt")
	return train_df, test_df


def clean_text_basic(text: str) -> str:
	text = text.lower().strip()
	text = re.sub(r"\s+", " ", text)
	return text


def add_clean_text(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df["clean_text"] = df["text"].apply(clean_text_basic)
	return df


def train_and_evaluate(
	data_dir: Path,
	results_path: Path,
	model_name: str = "bert-base-uncased",
	max_length: int = 128,
	epochs: int = 2,
) -> dict:
	train_df, test_df = load_data(data_dir)
	train_df = add_clean_text(train_df)
	test_df = add_clean_text(test_df)

	label_names = sorted(train_df["emotion"].unique())
	label_to_id = {label: idx for idx, label in enumerate(label_names)}
	id_to_label = {idx: label for label, idx in label_to_id.items()}

	train_df["label"] = train_df["emotion"].map(label_to_id)
	test_df["label"] = test_df["emotion"].map(label_to_id)

	train_dataset = Dataset.from_pandas(train_df[["clean_text", "label"]])
	test_dataset = Dataset.from_pandas(test_df[["clean_text", "label"]])

	tokenizer = AutoTokenizer.from_pretrained(model_name)

	def tokenize_batch(batch):
		return tokenizer(
			batch["clean_text"],
			padding="max_length",
			truncation=True,
			max_length=max_length,
		)

	train_dataset = train_dataset.map(tokenize_batch, batched=True)
	test_dataset = test_dataset.map(tokenize_batch, batched=True)

	train_dataset = train_dataset.remove_columns(["clean_text"])
	test_dataset = test_dataset.remove_columns(["clean_text"])
	train_dataset.set_format("torch")
	test_dataset.set_format("torch")

	model = AutoModelForSequenceClassification.from_pretrained(
		model_name,
		num_labels=len(label_names),
		id2label=id_to_label,
		label2id=label_to_id,
	)

	def compute_metrics(pred):
		labels = pred.label_ids
		preds = np.argmax(pred.predictions, axis=1)
		accuracy = accuracy_score(labels, preds)
		precision, recall, f1, _ = precision_recall_fscore_support(
			labels,
			preds,
			average="macro",
			zero_division=0,
		)
		return {
			"accuracy": accuracy,
			"precision": precision,
			"recall": recall,
			"f1": f1,
		}

	output_dir = Path(results_path).parent / "transformer_checkpoints"
	training_args = TrainingArguments(
		output_dir=str(output_dir),
		eval_strategy="epoch",
		save_strategy="epoch",
		learning_rate=2e-5,
		per_device_train_batch_size=16,
		per_device_eval_batch_size=16,
		num_train_epochs=epochs,
		weight_decay=0.01,
		logging_steps=50,
		load_best_model_at_end=True,
		metric_for_best_model="f1",
	)

	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		data_collator=data_collator,
		compute_metrics=compute_metrics,
	)

	trainer.train()
	eval_metrics = trainer.evaluate()

	metrics = {
		"accuracy": float(eval_metrics.get("eval_accuracy", 0.0)),
		"precision": float(eval_metrics.get("eval_precision", 0.0)),
		"recall": float(eval_metrics.get("eval_recall", 0.0)),
		"f1": float(eval_metrics.get("eval_f1", 0.0)),
	}
	save_metrics_row(metrics, "TRANSFORMER", results_path)

	val_path = Path(data_dir) / "validation.txt"
	if val_path.exists():
		val_df = load_split(val_path)
		val_df = add_clean_text(val_df)
		val_df["label"] = val_df["emotion"].map(label_to_id)
		val_dataset = Dataset.from_pandas(val_df[["clean_text", "label"]])
		val_dataset = val_dataset.map(tokenize_batch, batched=True)
		val_dataset = val_dataset.remove_columns(["clean_text"])
		val_dataset.set_format("torch")
		val_pred = trainer.predict(val_dataset)
		val_metrics = compute_metrics(val_pred)
		save_metrics_row(val_metrics, "TRANSFORMER_VAL", results_path)

	return metrics


def train(data_dir: Path, results_path: Path) -> dict:
	return train_and_evaluate(data_dir=data_dir, results_path=results_path)


def evaluate(data_dir: Path, results_path: Path) -> dict:
	return train_and_evaluate(data_dir=data_dir, results_path=results_path)
