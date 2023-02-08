"""
Field class
"""

from itertools import chain
import json
import numpy as np
import pickle
import time
from tqdm import tqdm

import torch
from plato.args import str2bool
from plato.data.tokenizer import Tokenizer


def max_lens(X):
    lens = [len(X)]
    while isinstance(X[0], list):
        lens.append(max(map(len, X)))
        X = [x for xs in X for x in xs]
    return lens


def list2np(X, padding=0, dtype="int64"):
    shape = max_lens(X)
    ret = np.full(shape, padding, dtype=np.int32)

    if len(shape) == 1:
        ret = np.array(X)
    elif len(shape) == 2:
        for i, x in enumerate(X):
            ret[i, :len(x)] = np.array(x)
    elif len(shape) == 3:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                ret[i, j, :len(x)] = np.array(x)
    return ret.astype(dtype)


class BPETextField(object):
    pad_token = "[PAD]"
    bos_token = "[BOS]"
    sep_token = "[SEP]"
    eos_token = "[EOS]"
    unk_token = "[UNK]"

    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("BPETextField")
        group.add_argument("--vocab_path", type=str, required=True,
                           help="The vocabulary file path.")
        group.add_argument("--filtered", type=str2bool, default=False,
                           help="Whether to filter the data with too long utterance/context. "
                                "If the data is unfiltered, it will be truncated.")
        group.add_argument("--max_len", type=int, default=256,
                           help="The maximum length of context or knowledges.")
        group.add_argument("--tokenizer_type", type=str, default="Bert",
                           choices=["Bert", "GPT2"],
                           help="The type of tokenizer.")
        return group

    def __init__(self, hparams):
        special_tokens = [self.pad_token, self.bos_token, self.sep_token, self.eos_token, self.unk_token]
        self.tokenizer = Tokenizer(vocab_path=hparams.vocab_path,
                                   special_tokens=special_tokens,
                                   tokenizer_type=hparams.tokenizer_type)

        self.filtered = hparams.filtered
        self.max_len = hparams.max_len
        # self.min_utt_len = hparams.min_utt_len
        # self.max_utt_len = hparams.max_utt_len
        # self.min_ctx_turn = hparams.min_ctx_turn
        # self.max_ctx_turn = hparams.max_ctx_turn - 1  # subtract reply turn

        self.img_len = (224 // 16) * (224 // 16)
        return

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def num_specials(self):
        return len(self.special_tokens)

    @property
    def pad_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.pad_token])[0]

    @property
    def bos_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.bos_token])[0]

    @property
    def sep_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sep_token])[0]

    @property
    def eos_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_token])[0]

    @property
    def unk_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.unk_token])[0]

    def numericalize(self, tokens):
        """
        here only "convert_tokens_to_ids",
        which need be tokenized into tokens(sub-words) by "tokenizer.tokenize" before
        """
        assert isinstance(tokens, list)
        if len(tokens) == 0:
            return []
        element = tokens[0]
        if isinstance(element, list):
            return [self.numericalize(s) for s in tokens]
        else:
            return self.tokenizer.convert_tokens_to_ids(tokens)

    def denumericalize(self, numbers):
        """
        here first "convert_ids_to_tokens", then combine sub-words into origin words
        """
        assert isinstance(numbers, list)
        if len(numbers) == 0:
            return []
        element = numbers[0]
        if isinstance(element, list):
            return [self.denumericalize(x) for x in numbers]
        else:
            return self.tokenizer.decode(
                numbers, ignore_tokens=[self.bos_token, self.eos_token, self.pad_token])

    def collate_fn_image_caption(self, samples, is_test=False):
        batch_size = len(samples)

        # Todo: preprocess the image
        src_token = np.stack([sample["src"].numpy() for sample in samples], axis=0)
        src_pos = np.zeros((batch_size, self.img_len), dtype="int64")
        src_pos[:] = np.arange(self.img_len, dtype="int64")
        src_mask = np.ones((batch_size, self.img_len))

        batch = {}
        batch["src_token"] = src_token
        batch["src_mask"] = src_mask  # 根据src_token生成mask，只有其中的0才全表示pad
        batch["src_pos"] = src_pos

        # Todo: preprocess the text
        if "tgt" in samples[0].keys():
            tgt = []
            for sample in samples:
                cap = sample["tgt"]
                cap = self.tokenizer.tokenize(cap.strip())
                cap = [self.bos_id] + self.numericalize(cap) + [self.eos_id]
                if not is_test:
                    cap = cap[:self.max_len + 2]
                tgt.append(cap)

            # Token ids & Label ids
            tgt_token = list2np(tgt, padding=self.pad_id)

            # Position ids
            tgt_pos = np.zeros_like(tgt_token)
            tgt_pos[:] = np.arange(tgt_token.shape[1], dtype=tgt_token.dtype)

            batch["tgt_token"] = tgt_token
            batch["tgt_mask"] = (tgt_token != self.pad_id).astype("int64")
            batch["tgt_pos"] = tgt_pos
        return batch, batch_size

    def collate_fn_unim_poem(self, samples, is_test=False):
        batch_size = len(samples)

        tgt = []
        tgt_pos = []
        tgt_turn = []
        for sample in samples:
            poem = sample["tgt"]
            poem = [self.tokenizer.tokenize(sentence.strip()) for sentence in poem]
            result_poem = [self.bos_id]
            result_turn = [0]
            result_pos = [0]
            for idx, sentence in enumerate(poem):
                new_poem = self.numericalize(sentence) + [self.sep_id]
                result_poem += new_poem
                if idx == 0:
                    result_pos += list(range(1, len(new_poem) + 1))
                else:
                    result_pos += list(range(len(new_poem)))
                result_turn += [idx] * len(new_poem)
            result_poem[-1] = self.eos_id
            if not is_test:
                result_poem = result_poem[:self.max_len + 2]
                result_pos = result_pos[:self.max_len + 2]
                result_turn = result_turn[:self.max_len + 2]
            tgt.append(result_poem)
            tgt_pos.append(result_pos)
            tgt_turn.append(result_turn)

        # Token ids & Label ids
        tgt_token = list2np(tgt, padding=self.pad_id)
        tgt_pos = list2np(tgt_pos, padding=self.pad_id)
        tgt_turn = list2np(tgt_turn, padding=self.pad_id)

        batch = {}
        batch["tgt_token"] = tgt_token
        batch["tgt_mask"] = (tgt_token != self.pad_id).astype("int64")
        batch["tgt_pos"] = tgt_pos
        batch["tgt_turn"] = tgt_turn

        return batch, batch_size

    def collate_fn_multim_poem(self, samples, is_test=False):
        batch_size = len(samples)

        # Todo: preprocess the image
        src_token = np.stack([sample["src"].numpy() for sample in samples], axis=0)
        src_pos = np.zeros((batch_size, self.img_len), dtype="int64")
        src_pos[:] = np.arange(self.img_len, dtype="int64")
        src_mask = np.ones((batch_size, self.img_len))

        batch = {}
        batch["src_token"] = src_token
        batch["src_mask"] = src_mask  # 根据src_token生成mask，只有其中的0才全表示pad
        batch["src_pos"] = src_pos

        # Todo: preprocess the text
        if "tgt" in samples[0].keys():
            tgt = []
            tgt_pos = []
            tgt_turn = []
            for sample in samples:
                poem = sample["tgt"]
                poem = [self.tokenizer.tokenize(sentence.strip()) for sentence in poem]
                result_poem = [self.bos_id]
                result_pos = [0]
                result_turn = [0]
                for idx, sentence in enumerate(poem):
                    new_poem = self.numericalize(sentence) + [self.sep_id]
                    result_poem += new_poem
                    if idx == 0:
                        result_pos += list(range(1, len(new_poem) + 1))
                    else:
                        result_pos += list(range(len(new_poem)))
                    result_turn += [idx] * len(new_poem)
                result_poem[-1] = self.eos_id
                if not is_test:
                    result_poem = result_poem[:self.max_len + 2]
                    result_pos = result_pos[:self.max_len + 2]
                    result_turn = result_turn[:self.max_len + 2]
                tgt.append(result_poem)
                tgt_pos.append(result_pos)
                tgt_turn.append(result_turn)

            # Token ids & Label ids
            tgt_token = list2np(tgt, padding=self.pad_id)
            tgt_pos = list2np(tgt_pos, padding=self.pad_id)
            tgt_turn = list2np(tgt_turn, padding=self.pad_id)

            batch["tgt_token"] = tgt_token
            batch["tgt_mask"] = (tgt_token != self.pad_id).astype("int64")
            batch["tgt_pos"] = tgt_pos
            batch["tgt_turn"] = tgt_turn
        return batch, batch_size
