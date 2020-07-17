from nltk.tokenize import wordpunct_tokenize
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np



PUNCTUATION = '!,-.:;?()'


def char_to_id(s):
    result = {}
    for i, c in enumerate(s):
        result[c] = i
    return result


PUNCT_TO_ID = char_to_id(' ' + PUNCTUATION)


def check_one(reference, hypothesis):
    correct = 0
    incorrect = 0
    ref = wordpunct_tokenize(reference)
    hyp = wordpunct_tokenize(hypothesis)
    ref_i, hyp_i = 0, 0
    punct_places = 0
    while ref_i < len(ref) and hyp_i < len(hyp):
        need_punct_check_ref = False
        need_punct_check_hyp = False
        cur_ref = ref[ref_i]
        if cur_ref in PUNCTUATION:
            need_punct_check_ref = True
            punct_places += 1
        cur_hyp = hyp[hyp_i]
        if cur_hyp in PUNCTUATION:
            need_punct_check_hyp = True
        if need_punct_check_ref and need_punct_check_hyp:
            if cur_ref == cur_hyp:
                correct += 1
            else:
                incorrect += 1
            ref_i += 1
            hyp_i += 1
            continue

        if need_punct_check_ref and not need_punct_check_hyp:
            incorrect += 1
            ref_i += 1
            continue

        if not need_punct_check_ref and need_punct_check_hyp:
            incorrect += 1
            hyp_i += 1
            continue

        assert cur_hyp == cur_ref, "The phrases are inconsistent!"
        ref_i += 1
        hyp_i += 1

    return correct/punct_places - incorrect/(2 * len(reference))


def calc_metrics(eval_pred):
    y_pred = np.argmax(eval_pred.predictions, axis=-1)
    y_true = eval_pred.label_ids
    interest_idx = np.logical_or(y_pred != 0, y_true != 0)

    result = {'acc': accuracy_score(y_true[interest_idx], y_pred[interest_idx])}
    for p, idx in PUNCT_TO_ID.items():
        result[p + '_roc_auc'] = roc_auc_score(y_true.flatten() == idx, eval_pred.predictions[:, :, idx].flatten())

    return result


