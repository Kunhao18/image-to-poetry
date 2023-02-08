"""
Metrics class.
"""

from collections import Counter

from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice


def coco_scores(refs, hyps):
    Bleu_metrics = Bleu(3)
    Rouge_metrics = Rouge()
    Cider_metrics = Cider()
    Meteor_metrics = Meteor()
    Spice_metrics = Spice()
    print("bleu", Bleu_metrics.compute_score(refs, hyps)[0])
    print("rouge", Rouge_metrics.compute_score(refs, hyps)[0])
    print("cider", Cider_metrics.compute_score(refs, hyps)[0])
    print("meteor", Meteor_metrics.compute_score(refs, hyps)[0])
    print("spice", Spice_metrics.compute_score(refs, hyps)[0])

    return 1


def coco_bleu(refs, hyps):
    Bleu_metrics = Bleu(3)
    return Bleu_metrics.compute_score(refs, hyps, verbose=0)[0]


def distinct(seqs):
    """ Calculate intra/inter distinct 1/2. """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2


def bleu(hyps, refs):
    """ Calculate bleu 1/2. """
    bleu_1 = []
    bleu_2 = []
    bleu_3 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.33, 0.33, 0.33, 0])
        except:
            score = 0
        bleu_3.append(score)
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    bleu_3 = np.average(bleu_3)
    return bleu_1, bleu_2, bleu_3


def novelty(hyps):
    poems = [" ".join(poem).split(" [SEP] ") for poem in hyps]
    for i in range(len(poems)):
        for j in range(len(poems[i])):
            poems[i][j] = poems[i][j].split()

    poem_num = len(poems)
    sent_nums = [len(poem) for poem in poems]
    max_sent_num = max(np.array(sent_nums))

    unigrams_novelty = 0
    bigrams_novelty = 0
    for i in range(max_sent_num):
        unigrams_all, bigrams_all = Counter(), Counter()
        for j in range(poem_num):
            if i < len(poems[j]):
                seq = poems[j][i]
                unigrams_all.update(Counter(seq))
                bigrams_all.update(Counter(zip(seq, seq[1:])))
        unigrams_novelty += (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
        bigrams_novelty += (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    unigrams_novelty /= max_sent_num
    bigrams_novelty /= max_sent_num

    return unigrams_novelty, bigrams_novelty

