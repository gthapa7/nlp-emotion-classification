# NLP Emotion Classification

Emotion classification on a six-class dataset using three model families: a fully connected baseline, a bidirectional LSTM, and a Transformer (BERT). The project includes notebooks for exploration and experimentation, plus a reusable `src` pipeline for training, evaluation, and metric logging.

## Project Structure

```
nlp-emotion-classification/
├── data/
│   ├── train.txt
│   └── test.txt
├── notebooks/
│   ├── Exploration.ipynb
│   ├── Preprocessing.ipynb
│   ├── Fc_model.ipynb
│   ├── Rnn_model.ipynb
│   ├── Transformer_model.ipynb
│   └── Comparison.ipynb
├── src/
│   ├── main.py
│   ├── evaluation.py
│   └── models/
│       ├── fc.py
│       ├── rnn.py
│       └── transformer.py
├── results/
│   ├── metrics.csv
│   ├── metrics_comparison.png
│   ├── training_efficiency.png
│   └── comprehensive_analysis.png
├── report/
│   └── final_report.md
└── README.md
```

## Notebooks

- **Exploration**: data inspection, label distribution, and text length analysis.
- **Preprocessing**: shared cleaning and tokenization approaches used across models.
- **Fc_model**: Bag-of-Words baseline with a dense network.
- **Rnn_model**: Embedding + bidirectional LSTM model.
- **Transformer_model**: BERT fine-tuning pipeline.
- **Comparison**: unified evaluation, charts, and conclusions.

## Source Pipeline

The `src` folder provides a reusable training and evaluation pipeline that mirrors the notebook logic.

Run a model:

```
python src/main.py --task train --model fc
python src/main.py --task train --model rnn
python src/main.py --task train --model transformer
```

Summarize best runs:

```
python src/main.py --task summary
```

## Evaluation Metrics

All models are evaluated using:

- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)

Metrics are appended to `results/metrics.csv` after each run.

## Results Snapshot

Key comparison charts are saved in `results/` and included in the comparison notebook.

## Requirements

The project relies on TensorFlow/Keras for FC and RNN, and Hugging Face Transformers for BERT. Install the usual dependencies used in the notebooks and the `src` pipeline.

## Validation Note

`data/validation.txt` is available and the only remaining step is to run evaluation on it for the final check. This provides an unbiased view of generalization without retraining on the validation data.
