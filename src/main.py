import argparse
from typing import Final

import torch
import yaml
from src.pipelines.model_service import ModelService
from src.pipelines.trainer import Trainer
from src.preprocessing.test_dataset import TestDataset
from torch import cuda

from src.preprocessing.test_graph_dataset import GraphDataset
from src.utils.logger import Logger

TRAIN_COMMAND: Final[str] = "train"
EVAL_COMMAND: Final[str] = "evaluate"
PREDICT_COMMAND: Final[str] = "predict"


def setup_parser() -> argparse.ArgumentParser:
    """
    Sets up the argument parser with the "config" and "pipeline" arguments.

    Returns:
        ArgumentParser: The argument parser with the added arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the configuration file with the specified pipeline options.",
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        required=True,
        choices=[TRAIN_COMMAND, PREDICT_COMMAND, EVAL_COMMAND],
        help="Pipeline to be started",
    )

    return parser


def main() -> None:
    """
    The main function of the script. It sets up the parser, reads the configuration files,
    and starts the training or prediction process based on the provided pipeline argument.

    Usage examples:
        python -m src_transformers.main --config data/test_configs/training_config.yaml --pipeline train
        python -m src_transformers.main -c data/test_configs/training_config.yaml -p train
    """
    parser = setup_parser()
    args, _ = parser.parse_known_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger = Logger()
    logger.write_text("Config", str(config))


    # Setting up GPU based on availability and usage preference
    gpu_activated = config.pop("use_gpu") and cuda.is_available()
    if gpu_activated:
        device = torch.device('cuda')
        print(
            f"Using the device '{cuda.get_device_name(device)}' for the started pipeline.")
    else:
        device = torch.device('cpu')
        print(
            "GPU was either deactivated or is not available, using the CPU for the started pipeline.")

    # Setting random seed for torch
    seed = config["training_parameters"]["seed"]
    if seed is not None:
        torch.manual_seed(seed)
        if device == torch.device("cuda"):
            torch.cuda.manual_seed(seed)


    model_parameters = config.pop("model_parameters")
    model_name, model_attributes = model_parameters.popitem()

    # Create instance for dataset and process if given in config
    process_dataset = config["dataset_parameters"].pop("process")
    dataset = GraphDataset(**config["dataset_parameters"])
    if process_dataset:
        dataset.process()

    if args.pipeline == TRAIN_COMMAND:
        model = ModelService.create_model(device=device,
                                          model_name=model_name,
                                          model_attributes=model_attributes)

        trainer = Trainer(
            dataset=dataset,
            model=model,
            device=device,
            logger=logger,
            **config.pop("training_parameters"))

        trainer.start_training()
        trainer.save_model()
    if args.pipeline == EVAL_COMMAND:
        pass
    elif args.pipeline == PREDICT_COMMAND:
        pass


if __name__ == "__main__":
    main()
