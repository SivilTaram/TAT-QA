import string
import re
from collections import Counter
from statistics import mean


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    if normalized_prediction == normalized_ground_truth:
        return 1
    else:
        return 0


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def read_file_f1(file_path):
    with open(file_path, "r", encoding="utf8") as read_f:
        lines = read_f.readlines()
        all_f1_scores = []
        all_em_scores = []
        for line in lines[1:]:
            _, predict, ground_truth, _ = line.split("\t")
            local_f1, local_pre, local_rec = f1_score(predict, ground_truth)
            all_f1_scores.append(local_f1)
            all_em_scores.append(exact_match_score(predict, ground_truth))
        print("Avg F1: {}, EM: {}".format(mean(all_f1_scores), mean(all_em_scores)))


if __name__ == '__main__':
    read_file_f1("results/bart_large.tsv")