from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from datetime import timedelta
import os

import argparse
import json
import os
import numpy as np

import base64
from io import BytesIO

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
from plato.metrics.metrics import bleu, distinct, novelty

app = Flask(__name__,
            template_folder="../vue/frontend/dist",
            static_folder="../vue/frontend/dist/static")

transforms_train = transforms.Compose([
    transforms.Resize(224),  # smaller edge of image resized to 224
    transforms.CenterCrop(224),  # get 224x224 crop from random location
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

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
Trainer.add_cmdline_argument(parser)
ModelBase.add_cmdline_argument(parser)
Generator.add_cmdline_argument(parser)

hparams = parse_args(parser)
hparams.use_gpu = torch.cuda.is_available() and hparams.gpu >= 0

if hparams.hparams_file and os.path.exists(hparams.hparams_file):
    print(f"Loading hparams from {hparams.hparams_file} ...")
    hparams.load(hparams.hparams_file)
    print(f"Loaded hparams from {hparams.hparams_file}")

bpe = BPETextField(hparams.BPETextField)
hparams.Model.num_token_embeddings = bpe.vocab_size
generator = Generator.create(hparams, bpe=bpe)
collate_fn = bpe.collate_fn_multim_poem


def to_tensor(array):
    """
    numpy array -> tensor
    """
    array = torch.tensor(array)
    return array.cuda() if hparams.use_gpu else array


model = ModelBase.create(hparams, generator=generator)
trainer = Trainer(model, to_tensor, hparams.Trainer)
trainer.load()

model.eval()


def parse_text(batch):
    """ Parse text. """
    return bpe.denumericalize(batch.tolist())


infer_parse_dict = {
    "tgt": parse_text,
    "preds": parse_text
}


def infer(image):
    with torch.no_grad():
        img = image.convert("RGB")
        img = transforms_train(img)
        samples = [{"src": img}]
        batch = collate_fn(samples, is_test=True)
        results = trainer.infer_single(batch[0], infer_parse_dict)
        scores = np.array(results[0]["scores"])
        chosen_idx = np.argmax(scores)
        chosen_poem = results[0]["preds"][chosen_idx]
        return chosen_poem


def initiate():
    img = Image.open("datasets/multim_poem/5.jpg")
    print(infer(img))


# 主页面
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


# 生成词云图片接口，以base64格式返回
@app.route('/poem/generate', methods=["POST"])
def generate():
    img_base64 = request.json.get("pic")
    image = base64.b64decode(img_base64)
    image = BytesIO(image)
    image = Image.open(image)
    res = infer(image)
    return res


if __name__ == '__main__':
    initiate()
    app.run(host="0.0.0.0")
