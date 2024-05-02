import argparse
from typing import Final

import torch
import yaml

from src.preprocessing.raw_data_generation import generate_raw_data
from src.utils.constants import DATASET_NAME_MAPPING
from src.utils.model_service import ModelService
from src.pipelines.trainer import Trainer
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
        print(f"[MAIN]: Using the device '{cuda.get_device_name(device)}' for the started pipeline.")
    else:
        device = torch.device('cpu')
        print("[MAIN]: GPU was either deactivated or is not available, using the CPU for the started pipeline.")

    # Setting random seed for torch
    seed = config["training_parameters"]["seed"]
    if seed is not None:
        torch.manual_seed(seed)
        if device == torch.device("cuda"):
            torch.cuda.manual_seed(seed)

    # Extracting model parameters from config
    model_parameters = config.pop("model_parameters")
    model_name, model_attributes = model_parameters.popitem()


    # TODO: clean up  raw params and dataset params
    # Create graph dataset
    # Extracting dataset parameters from config
    dataset_parameters = config.pop("dataset_parameters")
    dataset_attributes = dataset_parameters.pop("class_attributes")
    dataset_name = dataset_attributes.pop("class_name")
    process_dataset = dataset_parameters.pop("process")

    # Generate raw data file
    generate_raw_data(raw_file_name=dataset_attributes['raw_file_name'],
                      coordinate_transformation=config.pop('pre_processing_parameters')['coordinate_transformation'],
                      **dataset_parameters)

    # Create instance for dataset and process if given in config
    try:
        dataset = DATASET_NAME_MAPPING[dataset_name](**dataset_attributes)
    except KeyError as parse_error:
        raise (KeyError(f"The dataset '{dataset_name}' does not exist!")) from parse_error

    if process_dataset:
        dataset.process()

    if args.pipeline == TRAIN_COMMAND:
        model = ModelService.create_model(model_name=model_name,
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
