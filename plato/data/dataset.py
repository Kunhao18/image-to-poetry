"""
Dataset class
"""

import os
import json
from PIL import Image

import torch
from tqdm import tqdm

import queue


class Dataset(object):
    """ Basic Dataset interface class. """

    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("Dataset")
        return group

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ImageCaptionDataset(torch.utils.data.Dataset):
    """
    Lazy load dataset from disk.
    """

    def __init__(self, transforms, data_type):
        assert data_type in ['train', 'val'], "mode must be one of 'train' or 'val'."
        self.root = "datasets/caption"
        self.transforms = transforms
        self.data_type = data_type

        json_name = "captions_" + data_type + ".json"
        with open(os.path.join(self.root, json_name)) as f_in:
            json_file = json.load(f_in)
        self.imgs = {img_info["id"]: img_info["file_name"] for img_info in json_file["images"]}
        self.captions = json_file["annotations"]

    def __getitem__(self, idx):
        # load images
        caption = self.captions[idx]["caption"]
        caption = str(caption).lower()
        img_id = self.captions[idx]["image_id"]

        img_path = os.path.join(self.root, self.data_type, self.imgs[img_id])
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        return {"src": img, "tgt": caption}

    def __len__(self):
        return len(self.captions)


class UnimPoemDataset(torch.utils.data.Dataset):

    def __init__(self, transforms, data_type):
        super(UnimPoemDataset, self).__init__()
        assert data_type in ['train', 'val'], "mode must be one of 'train' or 'val'."
        self.root = "datasets"
        self.data_type = data_type

        self.poems = []
        file_name = "unim_poem_" + data_type + ".json"
        with open(os.path.join(self.root, file_name)) as f_in:
            json_file = json.load(f_in)
            for sample in tqdm(json_file):
                sentences = sample["poem"].strip().split("\n")
                # poem = " [EOS] ".join(sentences)
                # self.poems.append(poem)
                self.poems.append(sentences)

    def __getitem__(self, idx):
        poem = self.poems[idx]

        return {"tgt": poem}

    def __len__(self):
        return len(self.poems)


class MultiPoemDataset(torch.utils.data.Dataset):

    def __init__(self, transforms, data_type):
        super(MultiPoemDataset, self).__init__()
        assert data_type in ['train', 'val'], "mode must be one of 'train' or 'val'."
        self.root = "datasets"
        self.transforms = transforms
        self.data_type = data_type

        self.pairs = []
        file_name = "multim_poem_" + data_type + ".json"
        with open(os.path.join(self.root, file_name)) as f_in:
            json_file = json.load(f_in)
            for pair in tqdm(json_file):
                img_name = os.path.join(self.root, "multim_poem", str(pair["id"]) + ".jpg")
                # img = Image.open(img_name).convert("RGB")
                # img = self.transforms(img)
                sentences = pair["poem"].strip().split("\n")
                self.pairs.append((img_name, sentences))

    def __getitem__(self, idx):
        poem = self.pairs[idx][1]
        img_path = self.pairs[idx][0]
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        # img = self.transforms(self.pairs[idx][0])

        return {"src": img, "tgt": poem}

    def __len__(self):
        return len(self.pairs)


class MultiPoemMatchDataset(torch.utils.data.Dataset):

    def __init__(self, transforms):
        super(MultiPoemMatchDataset, self).__init__()
        self.root = "datasets"
        self.transforms = transforms

        self.pairs = []
        file_name = "multim_poem.json"
        with open(os.path.join(self.root, file_name)) as f_in:
            json_file = json.load(f_in)
            for pair in tqdm(json_file):
                img_name = os.path.join(self.root, "multim_poem", str(pair["id"]) + ".jpg")
                try:
                    img = Image.open(img_name).convert("RGB")
                    # img = self.transforms(img)
                    sentences = pair["poem"].strip().split("\n")
                    self.pairs.append((img, sentences, pair["id"]))
                except:
                    continue

    def __getitem__(self, idx):
        poem = self.pairs[idx][1]
        # img_path = self.pairs[idx][0]
        # img = Image.open(img_path).convert("RGB")
        # img = self.transforms(img)
        img = self.transforms(self.pairs[idx][0])
        img_id = self.pairs[idx][2]

        return {"src": img, "tgt": poem, "img_id": img_id}

    def __len__(self):
        return len(self.pairs)
