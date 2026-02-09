# Final Report: NLP Emotion Classification

## Objective
Build and compare three model families for six-class emotion classification: a fully connected baseline, a bidirectional LSTM, and a Transformer (BERT).

## Dataset
- Input: tweet-like text samples with emotion labels
- Splits: train/test provided

## Preprocessing
- Lowercasing and whitespace cleanup
- Bag-of-Words and TF-IDF for the FC baseline
- Tokenization + padding for the RNN
- Pretrained tokenizer for the Transformer

## Models
1. FC Baseline (BoW): dense network with dropout and L2 regularization
2. RNN (BiLSTM): embedding + bidirectional LSTM + dense classifier
3. Transformer (BERT): fine-tuned bert-base-uncased

## Evaluation Metrics
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)

## Results Summary
- FC (BoW): Accuracy 0.8885, F1 0.8417
- RNN (BiLSTM): Accuracy 0.8795, F1 0.8201
- Transformer (BERT): Accuracy 0.9240, F1 0.8831

## Key Findings
- The Transformer achieves the best macro F1 and overall accuracy.
- The FC baseline is a strong low-cost option.
- The RNN does not outperform the FC baseline on this dataset.

## Conclusion
For this task, the Transformer is the best performing model, while the FC baseline offers a strong and efficient alternative when compute is limited.
