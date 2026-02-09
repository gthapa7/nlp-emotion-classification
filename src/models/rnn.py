from pathlib import Path
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.evaluation import compute_metrics, save_metrics_row


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


def clean_text_alpha(text: str) -> str:
	text = text.lower().strip()
	text = re.sub(r"[^a-z\s]", " ", text)
	text = re.sub(r"\s+", " ", text)
	return text


def add_clean_text(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df["clean_text"] = df["text"].apply(clean_text_alpha)
	return df


def encode_labels(
	train_df: pd.DataFrame,
	test_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, list[str], LabelEncoder]:
	label_encoder = LabelEncoder()
	y_train = label_encoder.fit_transform(train_df["emotion"])
	y_test = label_encoder.transform(test_df["emotion"])
	class_names = list(label_encoder.classes_)
	return y_train, y_test, class_names, label_encoder


def build_model(max_words: int, max_len: int, num_classes: int) -> tf.keras.Model:
	embedding_dim = 100
	model = tf.keras.Sequential(
		[
			tf.keras.layers.Embedding(input_dim=max_words, output_dim=embedding_dim),
			tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
			tf.keras.layers.Dense(6, activation="relu"),
			tf.keras.layers.Dense(num_classes, activation="softmax"),
		]
	)
	model.compile(
		optimizer="adam",
		loss="categorical_crossentropy",
		metrics=["accuracy"],
	)
	model.build((None, max_len))
	return model


def train_and_evaluate(data_dir: Path, results_path: Path, epochs: int = 10) -> dict:
	train_df, test_df = load_data(data_dir)
	train_df = add_clean_text(train_df)
	test_df = add_clean_text(test_df)

	max_words = 20000
	text_tokenizer = Tokenizer(num_words=max_words, oov_token="<UNK>")
	text_tokenizer.fit_on_texts(train_df["clean_text"])

	train_seq = text_tokenizer.texts_to_sequences(train_df["clean_text"])
	test_seq = text_tokenizer.texts_to_sequences(test_df["clean_text"])

	seq_lengths = np.array([len(seq) for seq in train_seq])
	max_len = int(np.percentile(seq_lengths, 95))

	X_train = pad_sequences(train_seq, maxlen=max_len, padding="post", truncating="post")
	X_test = pad_sequences(test_seq, maxlen=max_len, padding="post", truncating="post")

	y_train, y_test, class_names, label_encoder = encode_labels(train_df, test_df)
	num_classes = len(class_names)
	y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

	model = build_model(max_words=max_words, max_len=max_len, num_classes=num_classes)
	early_stop = tf.keras.callbacks.EarlyStopping(
		monitor="val_loss",
		patience=2,
		restore_best_weights=True,
	)
	model.fit(
		X_train,
		y_train_cat,
		epochs=epochs,
		batch_size=32,
		validation_split=0.1,
		callbacks=[early_stop],
		verbose=1,
	)

	probs = model.predict(X_test, verbose=0)
	y_pred = probs.argmax(axis=1)
	metrics = compute_metrics(y_test, y_pred)
	save_metrics_row(metrics, "RNN_LSTM", results_path)

	val_path = Path(data_dir) / "validation.txt"
	if val_path.exists():
		val_df = load_split(val_path)
		val_df = add_clean_text(val_df)
		val_seq = text_tokenizer.texts_to_sequences(val_df["clean_text"])
		X_val = pad_sequences(val_seq, maxlen=max_len, padding="post", truncating="post")
		y_val = label_encoder.transform(val_df["emotion"])
		val_probs = model.predict(X_val, verbose=0)
		y_val_pred = val_probs.argmax(axis=1)
		val_metrics = compute_metrics(y_val, y_val_pred)
		save_metrics_row(val_metrics, "RNN_LSTM_VAL", results_path)

	return metrics


def train(data_dir: Path, results_path: Path) -> dict:
	return train_and_evaluate(data_dir=data_dir, results_path=results_path)


def evaluate(data_dir: Path, results_path: Path) -> dict:
	return train_and_evaluate(data_dir=data_dir, results_path=results_path)
