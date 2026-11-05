# Named Entity Recognition with a BiLSTM (PyTorch)

<p align="center">
  <b>A sequence-labeling NLP project that identifies people, organizations, places, geopolitical entities, dates, and other named entities in text.</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/Deep%20Learning-PyTorch-ee4c2c?logo=pytorch&logoColor=white">
  <img alt="Task" src="https://img.shields.io/badge/NLP-Named%20Entity%20Recognition-6f42c1">
  <img alt="Evaluation" src="https://img.shields.io/badge/Test%20Micro%20F1-0.82-success">
</p>

## Project overview

Named Entity Recognition (NER) is the NLP task of locating named entities in unstructured text and assigning each entity a category. This project builds an end-to-end, word-level NER system using **PyTorch and a Bidirectional Long Short-Term Memory network (BiLSTM)**.

The model reads a sentence in both directions. This gives each token representation access to information from the words before and after it, helping the classifier distinguish entities using sentence context rather than relying only on the token itself. A linear prediction layer then assigns one of 17 BIO tags to each token.

The notebook covers the complete experimental workflow: dataset inspection, sentence reconstruction, vocabulary creation, train/validation/test splitting, padded batching, model construction, training, checkpoint selection, entity-level evaluation, and inference on new sentences.

**Why this project matters:** extracting people, organizations, locations, and time expressions is a foundational capability for turning large volumes of text into structured information. NER can support downstream systems such as document search, news analysis, knowledge extraction, customer-support analytics, and information retrieval. These are potential applications of the technique; this repository demonstrates the model and evaluation workflow rather than a deployed production service.

## Results at a glance

| Project metric | Result |
|---|---:|
| Annotated dataset rows supplied | 1,048,575 |
| Sentences after preprocessing | 47,959 |
| Training sentences | 38,367 |
| Validation sentences | 4,796 |
| Test sentences | 4,796 |
| Vocabulary size | 18,802 (including `<PAD>` and `<UNK>`) |
| BIO output labels | 17 |
| Model parameters | 2,675,217 |
| Test micro F1 | **0.82** |
| Test weighted F1 | **0.82** |
| Test macro F1 | **0.64** |

The reported test metrics come from the notebook's `seqeval` classification report. They are **sequence/entity-level NER metrics**, not token accuracy. The macro score is lower than the micro and weighted scores, reflecting weaker performance on some less frequent entity categories.

## Model architecture

```text
Input sentence
      │
      ▼
Whitespace-tokenized words
      │
      ▼
Lowercase word-to-index lookup
( <PAD> = 0, <UNK> = 1 )
      │
      ▼
Learned word embeddings
18,802 vocabulary entries × 128 dimensions
      │
      ▼
Dropout (p = 0.3)
      │
      ▼
Bidirectional LSTM
128 hidden units per direction
      │
      ▼
Concatenate forward and backward states
256-dimensional token representation
      │
      ▼
Dropout (p = 0.3)
      │
      ▼
Linear token classifier
256 → 17 BIO tag scores
      │
      ▼
Argmax decoding
      │
      ▼
BIO tags → extracted entity spans
```

### Architecture details

| Component | Configuration |
|---|---|
| Embedding | Trainable `nn.Embedding`, 128 dimensions, padding index 0 |
| Recurrent encoder | One bidirectional LSTM layer in the saved checkpoint |
| Hidden size | 128 per direction; 256 after concatenation |
| Dropout modules | 0.3 before and after the BiLSTM |
| Classification head | Linear layer, 256 input features and 17 output classes |
| Decoding | Per-token `argmax` |
| Total parameters | 2,675,217 |

**Architecture clarification:** the notebook's model constructor exposes a `num_layers=2` argument, but it does not pass `num_layers` into `nn.LSTM`. PyTorch therefore uses its default of one recurrent layer. The saved checkpoint confirms this: it contains the first-layer (`l0`) forward and reverse LSTM weights, but no second-layer (`l1`) weights. This README describes the model that was actually trained and saved, not the intended constructor setting. The explicit dropout modules around the LSTM are active; the LSTM's built-in inter-layer dropout has no effect with a single recurrent layer.

This is a **BiLSTM + linear token-classification model**, not a BiLSTM-CRF. It predicts each token's tag with an independent linear classification step and does not use a CRF transition layer to enforce valid BIO tag sequences.

## Entity categories

The dataset uses BIO (Beginning, Inside, Outside) labels. `B-` marks the first token of an entity, `I-` marks a continuation token, and `O` marks a token outside any named entity.

| Entity label | Meaning |
|---|---|
| `PER` | Person |
| `ORG` | Organization |
| `GEO` | Geographical entity |
| `GPE` | Geopolitical entity |
| `TIM` | Time expression |
| `ART` | Artifact |
| `EVE` | Event |
| `NAT` | Natural phenomenon |
| `O` | Not part of a named entity |

Each entity type has a beginning and inside label, plus the shared `O` label: 8 entity types × 2 BIO prefixes + 1 outside label = **17 output classes**.

## Dataset and preprocessing

The supplied CSV uses the columns `Sentence #`, `Word`, `POS`, and `Tag`. Its format matches the GMB-derived *Annotated Corpus for Named Entity Recognition* commonly distributed through Kaggle. See the dataset reference in [Further reading](#further-reading).

The notebook performs the following preprocessing steps:

1. Reads the CSV using `latin1` encoding.
2. Forward-fills `Sentence #` so every token row is associated with its sentence.
3. Removes rows with missing `Word` or `Tag` values.
4. Groups tokens and BIO labels by sentence identifier.
5. Lowercases words for vocabulary lookup.
6. Builds a vocabulary from words occurring at least twice, reserving index `0` for `<PAD>` and index `1` for `<UNK>`.
7. Splits the sentences into training, validation, and test sets using an 80/10/10 split and `random_state=42`.
8. Pads each batch dynamically. Input padding uses `0`; label padding uses `-1`, which is ignored by the loss function.

The resulting vocabulary contains **18,802 entries**, and the dataset contains **47,959 grouped sentences**. Although the CSV includes a POS column, the current model uses only the `Word` and `Tag` columns; POS features are not fed into the network.

## Training configuration

| Setting | Value |
|---|---:|
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Loss | Cross-entropy (`ignore_index=-1`) |
| Training batch size | 32 |
| Validation/test batch size | 64 |
| Maximum epochs | 20 |
| Gradient clipping | Maximum norm 5.0 |
| Learning-rate scheduler | `ReduceLROnPlateau` (factor 0.5, patience 2) |
| Early stopping | 5 epochs without validation-F1 improvement |
| Selection criterion | Best validation entity-level F1 |
| Device selection | Apple MPS when available; otherwise CPU |

The best validation checkpoint is saved as `best_bilstm_ner.pt`. The notebook then reloads that checkpoint and evaluates it on the held-out test set.

## Test-set performance

The following values are copied from the notebook's `seqeval` classification report and rounded to two decimal places.

| Entity | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Artifact (`art`) | 0.33 | 0.02 | 0.04 | 48 |
| Event (`eve`) | 0.48 | 0.29 | 0.36 | 34 |
| Geography (`geo`) | 0.84 | 0.89 | 0.86 | 3,819 |
| Geopolitical entity (`gpe`) | 0.96 | 0.94 | 0.95 | 1,592 |
| Natural phenomenon (`nat`) | 0.67 | 0.62 | 0.64 | 13 |
| Organization (`org`) | 0.68 | 0.63 | 0.66 | 2,095 |
| Person (`per`) | 0.75 | 0.75 | 0.75 | 1,731 |
| Time (`tim`) | 0.88 | 0.87 | 0.87 | 1,990 |
| **Micro average** | **0.82** | **0.82** | **0.82** | **11,322** |
| **Macro average** | **0.70** | **0.63** | **0.64** | **11,322** |
| **Weighted average** | **0.82** | **0.82** | **0.82** | **11,322** |

### What the results indicate

- The strongest results are for geopolitical entities (`GPE`, F1 0.95), time expressions (`TIM`, F1 0.87), and geographical entities (`GEO`, F1 0.86).
- Person entities achieve an F1-score of 0.75, while organizations achieve 0.66.
- Rare categories are more challenging: the artifact class has an F1-score of 0.04 on 48 examples, and the event class has an F1-score of 0.36 on 34 examples. The natural-phenomenon score is based on only 13 examples, so it should be interpreted cautiously.
- The gap between the weighted F1 (0.82) and macro F1 (0.64) shows why reporting more than one average is important for an imbalanced NER dataset.

## Example predictions

The notebook includes a small inference demonstration using manually written sentences. Examples from that run:

**Input:** `Elon Musk founded Tesla in 2003`

| Extracted entity | Predicted type |
|---|---|
| Elon Musk | Person (`PER`) |
| 2003 | Time (`TIM`) |

**Input:** `The United Nations held a meeting in New York on Monday`

| Extracted entity | Predicted type |
|---|---|
| United Nations | Organization (`ORG`) |
| New York | Geography (`GEO`) |
| Monday | Time (`TIM`) |

These examples demonstrate the notebook's inference and entity-span extraction functions; they are illustrative predictions, not a separate benchmark.

## Repository structure

A clean GitHub repository can use the following layout:

```text
bilstm-named-entity-recognition/
│
├── BILSTM_NAME_ENTITY.ipynb   # Notebook: preprocessing, training, evaluation, inference
├── ner_dataset.csv            # Annotated token-level dataset
├── best_bilstm_ner.pt         # Saved PyTorch state_dict checkpoint
└── README.md                  # Project documentation
```

Rename the uploaded notebook to `BILSTM_NAME_ENTITY.ipynb` when adding it to the repository, if it still has a download suffix such as `(1)`.

## Run the project

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd bilstm-named-entity-recognition
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install torch numpy pandas scikit-learn seqeval matplotlib jupyter
```

For PyTorch installation options specific to your operating system and hardware, use the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Check the dataset path

The notebook currently contains a machine-specific absolute path in its CSV-loading cell. Replace it with a repository-relative path before running:

```python
df = pd.read_csv("ner_dataset.csv", encoding="latin1")
```

Keep `ner_dataset.csv` in the same directory as the notebook, or update the path to match your folder structure.

### 5. Run the notebook

```bash
jupyter notebook BILSTM_NAME_ENTITY.ipynb
```

Run the cells from top to bottom. The notebook preprocesses the data, creates the vocabulary and data loaders, trains the model, saves the best checkpoint, evaluates the test set, and demonstrates inference.

## Loading the saved checkpoint

The `.pt` file contains the model's **state dictionary**. The model architecture and the vocabulary/tag mappings must be recreated before loading it. Run the notebook cells that prepare `word2idx`, `tag2idx`, `idx2tag`, and `model`, then load the checkpoint:

```python
model.load_state_dict(
    torch.load("best_bilstm_ner.pt", map_location=device)
)
model.eval()
```

The current notebook rebuilds the vocabulary from the CSV instead of saving the mappings as separate artifacts. For reliable standalone inference, preserve the exact vocabulary and tag mappings alongside the model checkpoint in a future revision.

## Skills demonstrated

- **Natural Language Processing:** named entity recognition, sequence labeling, BIO tagging, entity-span extraction.
- **Deep Learning:** BiLSTM architecture, trainable embeddings, dropout, backpropagation, gradient clipping, checkpointing.
- **PyTorch engineering:** custom `Dataset`, `DataLoader`, dynamic batch padding, device-aware training and inference.
- **Data preparation:** handling missing sentence identifiers, grouping token sequences, vocabulary construction, unknown-token handling, and dataset splitting.
- **Model evaluation:** validation-based checkpoint selection and entity-level precision, recall, and F1 using `seqeval`.
- **Experiment workflow:** learning-rate scheduling, early stopping, test-set reporting, and qualitative inference examples.

## Limitations and next steps

This repository is a practical BiLSTM baseline. The following improvements would make it more robust and production-ready:

1. **Persist preprocessing artifacts:** save `word2idx`, `tag2idx`, and `idx2tag` with the model so inference does not depend on rebuilding mappings from the dataset.
2. **Correct and retrain the layer configuration:** pass `num_layers` explicitly to `nn.LSTM` if a stacked two-layer network is intended. This changes the architecture, so the existing single-layer checkpoint should not be presented as a two-layer model or loaded as though it were one.
3. **Fit vocabulary on training data only:** the current notebook constructs the vocabulary before the train/validation/test split. This uses token-frequency information from all partitions, although it does not use their labels. For a stricter evaluation protocol, build the vocabulary from the training partition and map unseen validation/test tokens to `<UNK>`.
4. **Add sequence-aware decoding:** evaluate a CRF layer to model dependencies between adjacent BIO labels and reduce invalid tag transitions.
5. **Handle padded sequences more carefully:** consider `pack_padded_sequence` so the recurrent network does not process right-padding as if it were part of each sentence.
6. **Improve tokenization and decoding:** replace simple whitespace splitting with a tokenizer consistent with the training data, and validate entity-type transitions while reconstructing spans.
7. **Address label imbalance:** experiment with class weighting, resampling, or additional labeled examples for rare categories such as artifacts and events.
8. **Use richer token representations:** compare the current learned word embeddings with pretrained embeddings or character/subword features, especially for unseen names and spelling variants.
9. **Strengthen reproducibility and evaluation:** set Python/NumPy/PyTorch random seeds, report multiple runs, perform per-class error analysis, and test on data from a different source or domain.
10. **Compare against modern baselines:** benchmark against BiLSTM-CRF and transformer-based NER models under the same data split and evaluation protocol.

## Further reading

- Lample et al. (2016), [*Neural Architectures for Named Entity Recognition*](https://arxiv.org/abs/1603.01360) — foundational work on neural NER architectures, including BiLSTM-CRF. The paper is useful background; this repository implements a BiLSTM with a linear token classifier rather than the paper's CRF decoder.
- [PyTorch `torch.nn.LSTM` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) — bidirectionality, recurrent layers, tensor shapes, and dropout behavior.
- [seqeval](https://github.com/chakki-works/seqeval) — sequence-labeling metrics used for the notebook's NER classification report.
- [Annotated Corpus for Named Entity Recognition (Kaggle)](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/) — reference dataset listing for the GMB-derived corpus format.
- [Domino: Building a named entity recognition model using a BiLSTM-CRF network](https://domino.ai/blog/named-entity-recognition-ner-challenges-and-model) — background on NER, IOB labels, the dataset fields, and the distinction between BiLSTM and BiLSTM-CRF architectures.

---

**Note:** Metrics and implementation details in this README are based on the supplied notebook, CSV, and saved checkpoint. Results should be re-evaluated after any changes to preprocessing, vocabulary construction, architecture, or dataset split.
