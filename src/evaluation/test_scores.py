import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")

def calc_accuracy(targets: torch.tensor, predictions: torch.tensor) -> dict[str, float]:
    """
    Calculate the accuracy of the model.

    Given the input of targets and predictions this function calculates the total accuracy and the separate accuracy for each class.

    :param targets: True labels
    :param predictions: Predicted labels
    :return: Dictionary of the total and separate accuracy for each class
    """
    # TODO doc string
    total = accuracy_score(predictions,
                           targets)

    predictions_np = predictions.numpy()
    targets_np = targets.numpy()

    accuracy_healthy = accuracy_score(targets_np, predictions_np, sample_weight=(targets_np == 0).astype(np.int32))
    accuracy_sclerotic = accuracy_score(targets_np, predictions_np, sample_weight=(targets_np == 1).astype(np.int32))
    accuracy_dead = accuracy_score(targets_np, predictions_np, sample_weight=(targets_np == 2).astype(np.int32))


    score_dict = {
            "0_total": total,
            "1_healty": accuracy_healthy,
            "2_sclerotic": accuracy_sclerotic,
            "3_dead": accuracy_dead
        }

    return score_dict


def calc_score(targets: torch.tensor,
               predictions: torch.tensor,
               metric: classmethod,
               averaging: str = "macro") -> dict[str, float]:
    """
    Calculate the accuracy, F1, precision, or recall score of the model.

    Given the input of metric this function calculates the total score and the separate scores for each class.

    :param targets: True labels
    :param predictions: Predicted labels
    :param metric: The metric to calculate the score
    :param averaging: The type of averaging to be used (marco or weighted)
    :return: Dictionary of the total and separate scores for each class
    """
    # TODO doc string
    total = metric(predictions,
                   targets,
                   average=averaging)

    separate = metric(predictions,
                      targets,
                      average=None)

    score_dict = {
        "0_total": total,
        "1_healty": separate[0],
        "2_sclerotic": separate[1],
        "3_dead": separate[2]
    }

    return score_dict


def calc_test_scores(targets: torch.tensor, predictions: torch.tensor) -> dict[dict[str, float]]:
    """
    Calculate the accuracy, F1, precision, and recall scores of the model.
    :param targets: True labels
    :param predictions: Predicted labels
    :return: A dict with a dict for each score containing the total and separate scores for each class
    """

    # TODO doc string
    scores = {
        "accuracy": calc_accuracy(targets, predictions),
        "f1_macro": calc_score(targets, predictions, f1_score),
        "f1_weighted": calc_score(targets, predictions, f1_score, "weighted"),
        "precision": calc_score(targets, predictions, precision_score),
        "recall": calc_score(targets, predictions, recall_score)
    }

    return scores
