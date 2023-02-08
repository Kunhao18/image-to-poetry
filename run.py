"""
Running scripts.
"""

import argparse
import json
import os
import numpy as np
import copy
import random

import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from plato.args import parse_args
from plato.args import str2bool
from plato.data.data_loader import DataLoader
from plato.data.dataset import Dataset
from plato.data.dataset import ImageCaptionDataset, UnimPoemDataset, MultiPoemDataset, MultiPoemMatchDataset
from plato.data.field import BPETextField
from plato.trainer import Trainer
from plato.models.model_base import ModelBase
from plato.models.generator import Generator
from plato.metrics.metrics import bleu, distinct, novelty, coco_scores, coco_bleu

torch.autograd.set_detect_anomaly(True)

# (Optional)  Amend the image transform below.
transforms_train = transforms.Compose([
    transforms.Resize(224),  # smaller edge of image resized to 224
    transforms.CenterCrop(224),  # get 224x224 crop from random location
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", type=str2bool, default=False,
                        help="Whether to run trainning.")
    parser.add_argument("--do_test", type=str2bool, default=False,
                        help="Whether to run evaluation on the test dataset.")
    parser.add_argument("--do_infer", type=str2bool, default=False,
                        help="Whether to run inference on the test dataset.")
    parser.add_argument("--num_infer_batches", type=int, default=None,
                        help="The number of batches need to infer.\n"
                             "Stay 'None': infer on entrie test dataset.")
    parser.add_argument("--hparams_file", type=str, default=None,
                        help="Loading hparams setting from file(.json format).")
    BPETextField.add_cmdline_argument(parser)
    # Dataset.add_cmdline_argument(parser)
    Trainer.add_cmdline_argument(parser)
    ModelBase.add_cmdline_argument(parser)
    Generator.add_cmdline_argument(parser)

    hparams = parse_args(parser)
    hparams.use_gpu = torch.cuda.is_available() and hparams.gpu >= 0

    if hparams.hparams_file and os.path.exists(hparams.hparams_file):
        print(f"Loading hparams from {hparams.hparams_file} ...")
        hparams.load(hparams.hparams_file)
        print(f"Loaded hparams from {hparams.hparams_file}")

    print(json.dumps(hparams, indent=2))

    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)
    hparams.save(os.path.join(hparams.save_dir, "hparams.json"))

    bpe = BPETextField(hparams.BPETextField)
    hparams.Model.num_token_embeddings = bpe.vocab_size

    generator = Generator.create(hparams, bpe=bpe)

    COLLATE_FN = [bpe.collate_fn_image_caption,
                  bpe.collate_fn_image_caption,
                  bpe.collate_fn_unim_poem,
                  bpe.collate_fn_multim_poem,
                  bpe.collate_fn_multim_poem]
    DATA_SET = [ImageCaptionDataset,
                ImageCaptionDataset,
                UnimPoemDataset,
                MultiPoemDataset,
                MultiPoemDataset]
    collate_fn = COLLATE_FN[hparams.task_id]
    data_set = DATA_SET[hparams.task_id]

    if hparams.do_train:
        train_dataset = data_set(transforms_train, "train")
        train_loader = DataLoader(train_dataset, hparams.Trainer, collate_fn=collate_fn, is_train=True)
        valid_dataset = data_set(transforms_train, "val")
        valid_loader = DataLoader(valid_dataset, hparams.Trainer, collate_fn=collate_fn, is_test=True)
    elif hparams.do_infer or hparams.do_test:
        pass
    else:
        raise

    def to_tensor(array):
        """
        numpy array -> tensor
        """
        array = torch.tensor(array)
        return array.cuda() if hparams.use_gpu else array

    # Construct Model
    model = ModelBase.create(hparams, generator=generator)
    if hparams.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, model.optimizer, opt_level=hparams.fp16_opt_level)
        model.optimizer = optimizer

    # Construct Trainer
    trainer = Trainer(model, to_tensor, hparams.Trainer)

    # Load model
    trainer.load()

    if hparams.do_train:
        # Training process
        for epoch in range(hparams.num_epochs):
            trainer.train_epoch(train_loader, valid_loader)

    if hparams.do_test:
        # Validation process
        # trainer.evaluate(test_loader, need_save=False)
        model.eval()
        with torch.no_grad():
            multim_poem_dataset = MultiPoemMatchDataset(transforms_train)
            all_img_encod = None
            all_pom_encod = None
            all_img_id = []
            for idx in tqdm(range(len(multim_poem_dataset))):
                now_sample = collate_fn([multim_poem_dataset[idx]])[0]
                now_sample = type(now_sample)(map(lambda kv: (kv[0], to_tensor(kv[1])), now_sample.items()))
                now_encod = model._forward_image_poem_match(now_sample, True)
                img_encod = now_encod["src_encod"]
                pom_encod = now_encod["tgt_encod"]
                all_img_id.append(multim_poem_dataset[idx]["img_id"])
                if idx == 0:
                    all_img_encod = img_encod
                    all_pom_encod = pom_encod
                else:
                    all_img_encod = torch.cat([all_img_encod, img_encod], dim=0)
                    all_pom_encod = torch.cat([all_pom_encod, pom_encod], dim=0)

            all_match = []
            for idx in tqdm(range(all_img_encod.shape[0])):
                all_score = None
                for pom_idx in range(all_pom_encod.shape[0]):
                    now_score = torch.dot(all_img_encod[idx], all_pom_encod[pom_idx]).unsqueeze(0)
                    if pom_idx == 0:
                        all_score = now_score
                    else:
                        all_score = torch.cat([all_score, now_score], dim=0)
                scores, tops = torch.topk(all_score, 3)
                new_idx = [index.item() for idx, index in enumerate(tops) if scores[idx].item() > 657.5]
                all_match.append(new_idx)

            with open("match.txt", "w") as f_out:
                for idx, sample in enumerate(all_match):
                    f_out.write(str(all_img_id[idx]))
                    for index in sample:
                        f_out.write("\t" + str(index))
                    f_out.write("\n")

    if hparams.do_infer:
        # Inference process
        model.eval()

        def split(xs, sep, pad):
            """ Split id list by separator. """
            out, o = [], []
            for x in xs:
                if x == pad:
                    continue
                if x != sep:
                    o.append(x)
                else:
                    if len(o) > 0:
                        out.append(list(o))
                        o = []
            if len(o) > 0:
                out.append(list(o))
            assert (all(len(o) > 0 for o in out))
            return out

        def parse_context(batch):
            """ Parse context. """
            # print(batch)
            return bpe.denumericalize([split(xs, bpe.eos_id, bpe.pad_id)
                                       for xs in batch.tolist()])

        def parse_text(batch):
            """ Parse text. """
            return bpe.denumericalize(batch.tolist())

        infer_parse_dict = {
            "tgt": parse_text,
            "preds": parse_context
        }

        with torch.no_grad():
            with open("datasets/multim_poem_val.json") as f_in:
                test_set = json.load(f_in)
            img_path_prefix = "datasets/multim_poem/"

            comp_set = {}
            with open("test_normal.json") as f_in:
                val_file = json.load(f_in)
            for sample in val_file:
                img_id = sample["image_id"]
                poem = sample["caption"].split()
                for idx, word in enumerate(poem):
                    if word == "sep":
                        poem[idx] = "[SEP]"
                comp_set[img_id] = poem

            reference = {}
            hypothesis = {}
            for idx, sample in enumerate(test_set):
                ref_poem = " [SEP] ".join(sample["poem"].split("\n"))
                img_id = sample["id"]
                comp_poem = " ".join(comp_set[img_id])
                reference[idx] = [ref_poem]
                hypothesis[idx] = [comp_poem]

            print(coco_bleu(reference, hypothesis))
            print(coco_scores(reference, hypothesis))


if __name__ == "__main__":
    main()
