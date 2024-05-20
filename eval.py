import argparse
import json


def eval_entities(sentence, ground_truth, prediction):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    ground_truth_dict = {}
    for entity in ground_truth:
        ground_truth_dict[entity["name"].lower()] = entity["type"]

    prediction_dict = {}
    for entity in prediction:
        prediction_dict[entity["name"].lower()] = entity["type"]

    for entity in prediction:
        if (
            entity["name"].lower() in ground_truth_dict
            and entity["type"] == ground_truth_dict[entity["name"].lower()]
        ):
            true_positives += 1
        else:
            #  print(f"Sentence: {sentence}")
            #  print(f"False positive: {entity}")
            #  print(f"Ground truth: {ground_truth_dict.get(entity['name'])}")
            false_positives += 1

    for entity in ground_truth:
        if entity["name"].lower() not in prediction_dict:
            #  print(f"Sentence: {sentence}")
            #  print(f"False negative: {entity}")
            #  print(prediction)
            false_negatives += 1

    return true_positives, false_positives, false_negatives


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", help="Path to the ground truth file")
    parser.add_argument("prediction", help="Path to the prediction file")
    args = parser.parse_args()

    ground_truth = {}

    with open(args.ground_truth, "r") as f:
        for line in f:
            data = json.loads(line)
            ground_truth[data["sentence"]] = data["entities"]

    tp = 0
    fp = 0
    fn = 0

    with open(args.prediction, "r") as f:
        for line in f:
            data = json.loads(line)
            sentence = data["sentence"]
            prediction = data["predicted_entities"]

            if type(prediction) is not list:
                prediction = []

            if sentence not in ground_truth:
                print(f"Error: sentence '{sentence}' not found in ground truth")
                continue

            true_positives, false_positives, false_negatives = eval_entities(
                sentence, ground_truth[sentence], prediction
            )

            tp += true_positives
            fp += false_positives
            fn += false_negatives

    print(f"TP: {tp}, FP: {fp}, FN: {fn}")

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
