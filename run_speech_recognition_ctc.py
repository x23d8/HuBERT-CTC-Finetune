#!/usr/bin/env python
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "datasets[audio] >= 1.18.0",
#     "torch >= 1.5",
#     "torchaudio",
#     "librosa",
#     "jiwer",
#     "evaluate",
# ]
# ///

"""Fine-tuning a 🤗 Transformers CTC model for automatic speech recognition"""

import functools
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field

import datasets
import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import DatasetDict, load_dataset, load_from_disk

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0.dev0")

require_version(
    "datasets>=1.18.0",
    "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt",
)


logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the feature encoder layers of the model."},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout ratio for the attention probabilities."},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout ratio for activations inside the fully connected layer."},
    )
    feat_proj_dropout: float = field(default=0.0, metadata={"help": "The dropout ratio for the projected features."})
    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": (
                "Probability of each feature vector along the time axis to be chosen as the start of the vector "
                "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature "
                "vectors will be masked along the time axis."
            )
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Probability of each feature vector along the feature axis to be chosen as the start of the vectorspan"
                " to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature"
                " bins will be masked along the time axis."
            )
        },
    )
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})
    ctc_loss_reduction: str | None = field(
        default="mean",
        metadata={"help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."},
    )
    ctc_zero_infinity: bool | None = field(
        default=False,
        metadata={
            "help": "Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly"
            " occur when the inputs are too short to be aligned to the targets."
        },
    )
    add_adapter: bool | None = field(
        default=False,
        metadata={
            "help": "Whether a convolutional attention network should be stacked on top of the Wav2Vec2Bert Encoder. Can be very"
            "useful to downsample the output length."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str | None = field(
        default=None,
        metadata={"help": "Path or name of the dataset (cf `load_dataset` method of the Datasets library). Not required when using --use_preextracted_features."}
    )
    dataset_config_name: str = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (cf `load_dataset` method of the Datasets library)."
        },
    )
    train_split_name: str = field(
        default="train+validation",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to "
                "'train+validation'"
            )
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'test'"
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
            )
        },
    )
    chars_to_ignore: list[str] | None = list_field(
        default=None,
        metadata={"help": "A list of characters to remove from the transcripts."},
    )
    eval_metrics: list[str] = list_field(
        default=["wer"],
        metadata={"help": "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Filter audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0,
        metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"},
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    unk_token: str = field(
        default="[UNK]",
        metadata={"help": "The unk token for the tokenizer"},
    )
    pad_token: str = field(
        default="[PAD]",
        metadata={"help": "The padding token for the tokenizer"},
    )
    word_delimiter_token: str = field(
        default="|",
        metadata={"help": "The word delimiter token for the tokenizer"},
    )
    phoneme_language: str | None = field(
        default=None,
        metadata={
            "help": (
                "The target language that should be used be"
                " passed to the tokenizer for tokenization. Note that"
                " this is only relevant if the model classifies the"
                " input audio to a sequence of phoneme sequences."
            )
        },
    )
    use_preextracted_features: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use pre-extracted encoder hidden states instead of raw audio. "
                "When set, the script will load an Arrow dataset with pre-computed features "
                "and only fine-tune the CTC head (lm_head)."
            )
        },
    )
    features_dataset_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to an Arrow dataset saved by extract_feature.py --save_arrow. "
                "Required when use_preextracted_features is True."
            )
        },
    )


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for processing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: AutoProcessor
    padding: bool | str = "longest"
    pad_to_multiple_of: int | None = None
    pad_to_multiple_of_labels: int | None = None
    feature_extractor_input_name: str | None = "input_values"

    def __call__(self, features: list[dict[str, list[int] | torch.Tensor]]) -> dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            {self.feature_extractor_input_name: feature[self.feature_extractor_input_name]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        return batch


# ---------------------------------------------------------------------------
# CTC head-only model wrapper — for pre-extracted features
# ---------------------------------------------------------------------------

class CTCHeadOnlyModel(nn.Module):
    """
    Lightweight wrapper that takes pre-extracted encoder hidden states
    and only runs  dropout → lm_head → CTC loss.

    Compatible with HuggingFace Trainer (returns dict with 'loss' and 'logits').
    """

    def __init__(self, ctc_model):
        super().__init__()
        # Extract only the parts we need from the full CTC model
        self.dropout = ctc_model.dropout if hasattr(ctc_model, "dropout") else nn.Identity()
        self.lm_head = ctc_model.lm_head
        self.config = ctc_model.config

    def forward(self, hidden_states, labels=None, attention_mask=None, **kwargs):
        hidden = self.dropout(hidden_states)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            # CTC loss
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

            # input_lengths from attention_mask or full length
            if attention_mask is not None:
                input_lengths = attention_mask.sum(dim=-1).long()
            else:
                input_lengths = torch.full(
                    (logits.shape[0],), logits.shape[1], dtype=torch.long, device=logits.device
                )

            # target_lengths: count non-padding tokens (not -100)
            target_lengths = (labels != -100).sum(dim=-1).long()

            # Replace -100 with pad_token_id for CTC loss computation
            labels_for_ctc = labels.clone()
            labels_for_ctc[labels_for_ctc == -100] = 0

            ctc_loss = nn.CTCLoss(
                blank=self.config.pad_token_id,
                reduction=getattr(self.config, "ctc_loss_reduction", "mean"),
                zero_infinity=getattr(self.config, "ctc_zero_infinity", False),
            )
            loss = ctc_loss(log_probs, labels_for_ctc, input_lengths, target_lengths)

        return {"loss": loss, "logits": logits}


@dataclass
class DataCollatorPreextracted:
    """
    Data collator for pre-extracted hidden states.
    Pads hidden_states (2D: seq_len × hidden_dim) and labels.
    """

    pad_token_id: int = 0
    hidden_dim: int = 1024

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        # Reconstruct hidden_states tensors from flattened lists
        hidden_states_list = []
        for f in features:
            shape = f["hidden_states_shape"]
            hs = torch.tensor(f["hidden_states"], dtype=torch.float32).reshape(shape[0], shape[1])
            hidden_states_list.append(hs)

        # Pad hidden_states to max seq_len
        max_len = max(hs.shape[0] for hs in hidden_states_list)
        padded_hs = []
        attention_masks = []
        for hs in hidden_states_list:
            seq_len = hs.shape[0]
            pad_len = max_len - seq_len
            if pad_len > 0:
                padded = torch.cat([hs, torch.zeros(pad_len, hs.shape[1])], dim=0)
            else:
                padded = hs
            padded_hs.append(padded)
            mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
            attention_masks.append(mask)

        batch_hs = torch.stack(padded_hs)        # (batch, max_len, hidden_dim)
        batch_mask = torch.stack(attention_masks).long()  # (batch, max_len)

        # Pad labels
        label_list = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        max_label_len = max(l.shape[0] for l in label_list)
        padded_labels = []
        for l in label_list:
            pad_len = max_label_len - l.shape[0]
            if pad_len > 0:
                padded = torch.cat([l, torch.full((pad_len,), -100, dtype=torch.long)])
            else:
                padded = l
            padded_labels.append(padded)
        batch_labels = torch.stack(padded_labels)  # (batch, max_label_len)

        return {
            "hidden_states": batch_hs,
            "attention_mask": batch_mask,
            "labels": batch_labels,
        }


def create_vocabulary_from_data(
    datasets: DatasetDict,
    word_delimiter_token: str | None = None,
    unk_token: str | None = None,
    pad_token: str | None = None,
):
    # Given training and test labels create vocabulary
    def extract_all_chars(batch):
        all_text = " ".join(batch["target_text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = datasets.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=datasets["train"].column_names,
    )

    # take union of all unique characters in each dataset
    vocab_set = functools.reduce(
        lambda vocab_1, vocab_2: set(vocab_1["vocab"][0]) | set(vocab_2["vocab"][0]),
        vocabs.values(),
    )

    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_set))}

    # replace white space with delimiter token
    if word_delimiter_token is not None:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    return vocab_dict


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_process_index) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_process_index):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # =====================================================================
    # PRE-EXTRACTED FEATURES BRANCH (early exit)
    # When use_preextracted_features is True, we skip dataset loading
    # entirely and load the Arrow dataset from features_dataset_path.
    # =====================================================================
    if data_args.use_preextracted_features:
        if data_args.features_dataset_path is None:
            raise ValueError(
                "--features_dataset_path must be set when --use_preextracted_features is True."
            )

        logger.info("=" * 60)
        logger.info("USING PRE-EXTRACTED FEATURES MODE")
        logger.info(f"Loading features from: {data_args.features_dataset_path}")
        logger.info("Only the CTC head (lm_head + dropout) will be trained.")
        logger.info("=" * 60)

        # Load config
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            token=data_args.token,
            trust_remote_code=data_args.trust_remote_code,
        )

        # Load feature extractor (needed for saving processor later)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            token=data_args.token,
            trust_remote_code=data_args.trust_remote_code,
        )

        # Load Arrow dataset
        features_ds = load_from_disk(data_args.features_dataset_path)
        logger.info(f"Loaded dataset splits: {list(features_ds.keys())}")
        for split_name, ds in features_ds.items():
            logger.info(f"  {split_name}: {len(ds)} samples, columns: {ds.column_names}")

        # Text preprocessing: lowercasing + remove special chars
        chars_to_ignore_regex = (
            f"[{''.join(data_args.chars_to_ignore)}]" if data_args.chars_to_ignore is not None else None
        )

        def preprocess_text(batch):
            text = batch["text"]
            if chars_to_ignore_regex is not None:
                text = re.sub(chars_to_ignore_regex, "", text)
            batch["target_text"] = text.lower() + " "
            return batch

        features_ds = features_ds.map(preprocess_text, desc="preprocess text", load_from_cache_file=False)

        # Build vocabulary
        word_delimiter_token = data_args.word_delimiter_token
        unk_token = data_args.unk_token
        pad_token = data_args.pad_token
        tokenizer_name_or_path_feat = model_args.tokenizer_name_or_path
        tokenizer_kwargs_feat = {}

        if tokenizer_name_or_path_feat is None:
            tokenizer_name_or_path_feat = training_args.output_dir
            vocab_file = os.path.join(tokenizer_name_or_path_feat, "vocab.json")
            with training_args.main_process_first():
                if os.path.isfile(vocab_file):
                    try:
                        os.remove(vocab_file)
                    except OSError:
                        pass
            with training_args.main_process_first(desc="vocabulary creation"):
                if not os.path.isfile(vocab_file):
                    os.makedirs(tokenizer_name_or_path_feat, exist_ok=True)
                    vocab_dict = create_vocabulary_from_data(
                        features_ds,
                        word_delimiter_token=word_delimiter_token,
                        unk_token=unk_token,
                        pad_token=pad_token,
                    )
                    with open(vocab_file, "w") as file:
                        json.dump(vocab_dict, file)
            tokenizer_kwargs_feat = {
                "config": config,
                "tokenizer_type": config.model_type,
                "unk_token": unk_token,
                "pad_token": pad_token,
                "word_delimiter_token": word_delimiter_token,
            }

        tokenizer_feat = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path_feat,
            token=data_args.token,
            trust_remote_code=data_args.trust_remote_code,
            **tokenizer_kwargs_feat,
        )

        # Adapt config and create model
        config.update({
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": model_args.layerdrop,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "ctc_zero_infinity": model_args.ctc_zero_infinity,
            "pad_token_id": tokenizer_feat.pad_token_id,
            "vocab_size": len(tokenizer_feat),
            "activation_dropout": model_args.activation_dropout,
            "add_adapter": model_args.add_adapter,
        })

        model = AutoModelForCTC.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            config=config,
            token=data_args.token,
            trust_remote_code=data_args.trust_remote_code,
            ignore_mismatched_sizes=True,
        )

        # Rebuild lm_head if vocab size changed
        if model.lm_head.out_features != len(tokenizer_feat):
            logger.info(
                f"Rebuilding lm_head: {model.lm_head.out_features} → {len(tokenizer_feat)} classes"
            )
            model.lm_head = nn.Linear(config.hidden_size, len(tokenizer_feat))

        # Tokenize target text
        phoneme_language = data_args.phoneme_language

        def tokenize_targets(batch):
            additional_kwargs = {}
            if phoneme_language is not None:
                additional_kwargs["phonemizer_lang"] = phoneme_language
            batch["labels"] = tokenizer_feat(batch["target_text"], **additional_kwargs).input_ids
            return batch

        features_ds = features_ds.map(tokenize_targets, desc="tokenize targets", load_from_cache_file=False)

        # Limit samples
        if data_args.max_train_samples is not None and "train" in features_ds:
            n = min(data_args.max_train_samples, len(features_ds["train"]))
            features_ds["train"] = features_ds["train"].select(range(n))
        if data_args.max_eval_samples is not None and "eval" in features_ds:
            n = min(data_args.max_eval_samples, len(features_ds["eval"]))
            features_ds["eval"] = features_ds["eval"].select(range(n))

        # Get hidden_dim from first sample
        first_shape = features_ds["train"][0]["hidden_states_shape"]
        hidden_dim = first_shape[1]
        logger.info(f"Hidden dimension: {hidden_dim}")

        # Create CTC head-only model
        head_model = CTCHeadOnlyModel(model)
        head_model.to(training_args.device)

        # Data collator
        data_collator_feat = DataCollatorPreextracted(
            pad_token_id=tokenizer_feat.pad_token_id,
            hidden_dim=hidden_dim,
        )

        # Metrics
        eval_metrics_feat = {
            metric: evaluate.load(metric, cache_dir=model_args.cache_dir)
            for metric in data_args.eval_metrics
        }

        def preprocess_logits_feat(logits, labels):
            pred_ids = torch.argmax(logits, dim=-1)
            return pred_ids, labels

        _latest_predictions_feat = {"pred_str": [], "label_str": []}

        def compute_metrics_feat(pred):
            pred_ids = pred.predictions[0]
            pred.label_ids[pred.label_ids == -100] = tokenizer_feat.pad_token_id
            pred_str = tokenizer_feat.batch_decode(pred_ids)
            label_str = tokenizer_feat.batch_decode(pred.label_ids, group_tokens=False)

            logger.info("=" * 60)
            logger.info("EVALUATION PREDICTIONS vs GROUND TRUTH")
            logger.info("=" * 60)
            for i, (p, t) in enumerate(zip(pred_str, label_str)):
                logger.info(f"[Sample {i+1}]  PRED: {p}")
                logger.info(f"              TRUE: {t}")
            logger.info("=" * 60)

            _latest_predictions_feat["pred_str"] = list(pred_str)
            _latest_predictions_feat["label_str"] = list(label_str)

            metrics = {
                k: v.compute(predictions=pred_str, references=label_str)
                for k, v in eval_metrics_feat.items()
            }
            return metrics

        class EvalLoggingCallbackFeat(TrainerCallback):
            def __init__(self):
                self.all_eval_results = []

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics is not None:
                    epoch = state.epoch if state.epoch is not None else state.global_step
                    samples = []
                    for p, t in zip(
                        _latest_predictions_feat.get("pred_str", []),
                        _latest_predictions_feat.get("label_str", []),
                    ):
                        samples.append({"prediction": p, "ground_truth": t})
                    result_entry = {
                        "epoch": epoch,
                        "global_step": state.global_step,
                        **metrics,
                        "samples": samples,
                    }
                    self.all_eval_results.append(result_entry)
                    logger.info(f"EVAL — Epoch {epoch}, Step {state.global_step}")
                    for k, v in metrics.items():
                        logger.info(f"  {k}: {v}")

        eval_cb_feat = EvalLoggingCallbackFeat()

        # Save tokenizer / config
        with training_args.main_process_first():
            if is_main_process(training_args.local_process_index):
                feature_extractor.save_pretrained(training_args.output_dir)
                tokenizer_feat.save_pretrained(training_args.output_dir)
                config.save_pretrained(training_args.output_dir)

        # Disable auto-removal of unused columns — the DataCollator needs
        # hidden_states_shape which isn't in the model's forward() signature.
        training_args.remove_unused_columns = False

        # Trainer
        trainer_feat = Trainer(
            model=head_model,
            data_collator=data_collator_feat,
            args=training_args,
            compute_metrics=compute_metrics_feat,
            train_dataset=features_ds.get("train"),
            eval_dataset=features_ds.get("eval"),
            preprocess_logits_for_metrics=preprocess_logits_feat,
            callbacks=[eval_cb_feat],
        )

        # Train
        if training_args.do_train:
            train_result = trainer_feat.train()
            trainer_feat.save_model()
            metrics = train_result.metrics
            metrics["train_samples"] = len(features_ds["train"])
            trainer_feat.log_metrics("train", metrics)
            trainer_feat.save_metrics("train", metrics)
            trainer_feat.save_state()

        # Eval
        if training_args.do_eval and "eval" in features_ds:
            logger.info("*** Evaluate (pre-extracted features) ***")
            metrics = trainer_feat.evaluate()
            metrics["eval_samples"] = len(features_ds["eval"])
            trainer_feat.log_metrics("eval", metrics)
            trainer_feat.save_metrics("eval", metrics)

        # Save eval results JSON
        if eval_cb_feat.all_eval_results:
            path = os.path.join(training_args.output_dir, "all_eval_results.json")
            with open(path, "w") as f:
                json.dump(eval_cb_feat.all_eval_results, f, indent=2)
            logger.info(f"All eval results saved to {path}")

        logger.info("Done! (pre-extracted features mode)")
        return {}

    # =====================================================================
    # STANDARD AUDIO PIPELINE (requires dataset_name)
    # =====================================================================
    if data_args.dataset_name is None:
        raise ValueError(
            "--dataset_name is required when not using --use_preextracted_features."
        )

    # 1. First, let's load the dataset
    raw_datasets = DatasetDict()

    # Detect available splits in the dataset
    from datasets import get_dataset_split_names

    try:
        available_splits = get_dataset_split_names(
            data_args.dataset_name,
            data_args.dataset_config_name,
            trust_remote_code=data_args.trust_remote_code,
        )
        logger.info(f"Available splits for dataset '{data_args.dataset_name}': {available_splits}")
    except Exception as e:
        logger.warning(f"Could not detect available splits: {e}. Will proceed with provided split names.")
        available_splits = None

    # Resolve train split name: if the default "train+validation" is used but
    # "validation" doesn't exist, fall back to "train" only.
    train_split_name = data_args.train_split_name
    if "+" in train_split_name:
        requested_splits = [s.strip() for s in train_split_name.split("+")]
        if available_splits is not None:
            valid_splits = [s for s in requested_splits if s in available_splits]
            missing_splits = [s for s in requested_splits if s not in available_splits]
        else:
            # available_splits detection failed; probe each split individually
            valid_splits = []
            missing_splits = []
            for s in requested_splits:
                try:
                    load_dataset(
                        data_args.dataset_name,
                        data_args.dataset_config_name,
                        split=s,
                        token=data_args.token,
                        trust_remote_code=data_args.trust_remote_code,
                        streaming=True,
                    )
                    valid_splits.append(s)
                except Exception:
                    missing_splits.append(s)
        if missing_splits:
            logger.warning(
                f"Requested train splits {missing_splits} not found in dataset. "
                f"Available splits: {available_splits}. "
                f"Falling back to: {'+'.join(valid_splits) if valid_splits else 'train'}"
            )
        train_split_name = "+".join(valid_splits) if valid_splits else "train"

    # Resolve eval split name: fall back through "test" -> "validation" -> first non-train split
    eval_split_name = data_args.eval_split_name
    if available_splits is not None and eval_split_name not in available_splits:
        fallback_eval = None
        for candidate in ["test", "validation"]:
            if candidate in available_splits:
                fallback_eval = candidate
                break
        if fallback_eval is None:
            # Use the first split that isn't "train"
            non_train = [s for s in available_splits if s != "train"]
            fallback_eval = non_train[0] if non_train else "train"
        logger.warning(
            f"Eval split '{eval_split_name}' not found. "
            f"Available splits: {available_splits}. Falling back to '{fallback_eval}'."
        )
        eval_split_name = fallback_eval

    if training_args.do_train:
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=train_split_name,
            token=data_args.token,
            trust_remote_code=data_args.trust_remote_code,
        )

        if data_args.audio_column_name not in raw_datasets["train"].column_names:
            raise ValueError(
                f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'."
                " Make sure to set `--audio_column_name` to the correct audio column - one of"
                f" {', '.join(raw_datasets['train'].column_names)}."
            )

        if data_args.text_column_name not in raw_datasets["train"].column_names:
            raise ValueError(
                f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
                "Make sure to set `--text_column_name` to the correct text column - one of "
                f"{', '.join(raw_datasets['train'].column_names)}."
            )

        if data_args.max_train_samples is not None:
            max_train_samples = min(data_args.max_train_samples, len(raw_datasets["train"]))
            raw_datasets["train"] = raw_datasets["train"].select(range(max_train_samples))

    if training_args.do_eval:
        raw_datasets["eval"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=eval_split_name,
            token=data_args.token,
            trust_remote_code=data_args.trust_remote_code,
        )

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(data_args.max_eval_samples, len(raw_datasets["eval"]))
            raw_datasets["eval"] = raw_datasets["eval"].select(range(max_eval_samples))

    # 2. We remove some special characters from the datasets
    # that make training complicated and do not help in transcribing the speech
    # E.g. characters, such as `,` and `.` do not really have an acoustic characteristic
    # that could be easily picked up by the model
    chars_to_ignore_regex = (
        f"[{''.join(data_args.chars_to_ignore)}]" if data_args.chars_to_ignore is not None else None
    )
    text_column_name = data_args.text_column_name

    def remove_special_characters(batch):
        if chars_to_ignore_regex is not None:
            batch["target_text"] = re.sub(chars_to_ignore_regex, "", batch[text_column_name]).lower() + " "
        else:
            batch["target_text"] = batch[text_column_name].lower() + " "
        return batch

    with training_args.main_process_first(desc="dataset map special characters removal"):
        raw_datasets = raw_datasets.map(
            remove_special_characters,
            remove_columns=[text_column_name],
            desc="remove special characters from datasets",
        )

    # save special tokens for tokenizer
    word_delimiter_token = data_args.word_delimiter_token
    unk_token = data_args.unk_token
    pad_token = data_args.pad_token

    # 3. Next, let's load the config as we might need it to create
    # the tokenizer
    # load config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # 4. Next, if no tokenizer file is defined,
    # we create the vocabulary of the model by extracting all unique characters from
    # the training and evaluation datasets
    # We need to make sure that only first rank saves vocabulary
    # make sure all processes wait until vocab is created
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    tokenizer_kwargs = {}
    if tokenizer_name_or_path is None:
        # save vocab in training output dir
        tokenizer_name_or_path = training_args.output_dir

        vocab_file = os.path.join(tokenizer_name_or_path, "vocab.json")

        with training_args.main_process_first():
            if os.path.isfile(vocab_file):
                try:
                    os.remove(vocab_file)
                except OSError:
                    # in shared file-systems it might be the case that
                    # two processes try to delete the vocab file at the some time
                    pass

        with training_args.main_process_first(desc="dataset map vocabulary creation"):
            if not os.path.isfile(vocab_file):
                os.makedirs(tokenizer_name_or_path, exist_ok=True)
                vocab_dict = create_vocabulary_from_data(
                    raw_datasets,
                    word_delimiter_token=word_delimiter_token,
                    unk_token=unk_token,
                    pad_token=pad_token,
                )

                # save vocab dict to be loaded into tokenizer
                with open(vocab_file, "w") as file:
                    json.dump(vocab_dict, file)

        tokenizer_kwargs = {
            "config": config,
            "tokenizer_type": config.model_type,
            "unk_token": unk_token,
            "pad_token": pad_token,
            "word_delimiter_token": word_delimiter_token,
        }

    # 5. Now we can instantiate the feature extractor, tokenizer and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    # load feature_extractor and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        **tokenizer_kwargs, 
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # adapt config
    config.update(
        {
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": model_args.layerdrop,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "ctc_zero_infinity": model_args.ctc_zero_infinity,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
            "activation_dropout": model_args.activation_dropout,
            "add_adapter": model_args.add_adapter,
        }
    )

    # create model
    model = AutoModelForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )

    # freeze encoder
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # 6. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`

    # make sure that dataset decodes audio with correct sampling rate
    dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        raw_datasets = raw_datasets.cast_column(
            data_args.audio_column_name,
            datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
        )

    # derive max & min input length for sample rate & max duration
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    feature_extractor_input_name = feature_extractor.model_input_names[0]

    # `phoneme_language` is only relevant if the model is fine-tuned on phoneme classification
    phoneme_language = data_args.phoneme_language

    # Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    def prepare_dataset(batch):
        # load audio
        sample = batch[audio_column_name]

        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch[feature_extractor_input_name] = getattr(inputs, feature_extractor_input_name)[0]
        # take length of raw audio waveform
        batch["input_length"] = len(sample["array"].squeeze())

        # encode targets
        additional_kwargs = {}
        if phoneme_language is not None:
            additional_kwargs["phonemizer_lang"] = phoneme_language

        batch["labels"] = tokenizer(batch["target_text"], **additional_kwargs).input_ids
        return batch

    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=num_workers,
            desc="preprocess datasets",
        )

        def is_audio_in_length_range(length):
            return length > min_input_length and length < max_input_length

        # filter data that is shorter than min_input_length
        vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["input_length"],
        )

    # 7. Next, we can prepare the training.
    # Let's use word error rate (WER) as our evaluation metric,
    # instantiate a data collator and the trainer

    # Define evaluation metrics during training, *i.e.* word error rate, character error rate
    eval_metrics = {metric: evaluate.load(metric, cache_dir=model_args.cache_dir) for metric in data_args.eval_metrics}

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with ``args.preprocessing_only`` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step ``args.preprocessing_only`` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")
        return

    # For languages like Chinese with large vocabulary size, we need to discard logits
    # and only keep the argmax, otherwise we run out of memory during evaluation.
    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels

    # Store latest predictions for saving to JSON
    _latest_predictions = {"pred_str": [], "label_str": []}

    def compute_metrics(pred):
        pred_ids = pred.predictions[0]
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        # Log each prediiction vs ground truth
        logger.info("=" * 60)
        logger.info("EVALUATION PREDICTIONS vs GROUND TRUTH")
        logger.info("=" * 60)
        for i, (pred_text, label_text) in enumerate(zip(pred_str, label_str)):
            logger.info(f"[Sample {i+1}]")
            logger.info(f"  PRED: {pred_text}")
            logger.info(f"  TRUE: {label_text}")
        logger.info("=" * 60)

        # Save predictions for JSON export
        _latest_predictions["pred_str"] = list(pred_str)
        _latest_predictions["label_str"] = list(label_str)

        metrics = {k: v.compute(predictions=pred_str, references=label_str) for k, v in eval_metrics.items()}

        return metrics

    # Custom callback to log eval results per epoch and accumulate them
    class EvalLoggingCallback(TrainerCallback):
        def __init__(self):
            self.all_eval_results = []

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is not None:
                epoch = state.epoch if state.epoch is not None else state.global_step
                # Build per-sample predictions list
                samples = []
                for p, t in zip(_latest_predictions.get("pred_str", []), _latest_predictions.get("label_str", [])):
                    samples.append({"prediction": p, "ground_truth": t})
                result_entry = {
                    "epoch": epoch,
                    "global_step": state.global_step,
                    **metrics,
                    "samples": samples,
                }
                self.all_eval_results.append(result_entry)
                logger.info("=" * 60)
                logger.info(f"EVAL RESULTS — Epoch {epoch}, Step {state.global_step}")
                for k, v in metrics.items():
                    logger.info(f"  {k}: {v}")
                logger.info("=" * 60)

    eval_logging_callback = EvalLoggingCallback()

    # Now save everything to be able to create a single processor later
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_process_index):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)

    # Instantiate custom data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, feature_extractor_input_name=feature_extractor_input_name
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        processing_class=processor,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[eval_logging_callback],
    )

    # 8. Finally, we can start training

    # Training
    if training_args.do_train:
        # use last checkpoint if exist
        if os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save all accumulated eval results to JSON
    if eval_logging_callback.all_eval_results:
        all_eval_results_path = os.path.join(training_args.output_dir, "all_eval_results.json")
        with open(all_eval_results_path, "w") as f:
            json.dump(eval_logging_callback.all_eval_results, f, indent=2)
        logger.info(f"All eval results saved to {all_eval_results_path}")
    
    # Write model card and (optionally) push to hub
    config_name = data_args.dataset_config_name if data_args.dataset_config_name is not None else "na"
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
        "tags": ["automatic-speech-recognition", data_args.dataset_name],
        "dataset_args": (
            f"Config: {config_name}, Training split: {data_args.train_split_name}, Eval split:"
            f" {data_args.eval_split_name}"
        ),
        "dataset": f"{data_args.dataset_name.upper()} - {config_name.upper()}",
    }
    if "common_voice" in data_args.dataset_name:
        kwargs["language"] = config_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        try:
            trainer.create_model_card(**kwargs)
        except Exception as e:
            logger.warning(f"Could not create model card: {e}. Skipping model card creation.")

    return results


if __name__ == "__main__":
    main()