import argparse
from datetime import datetime
from typing import Final

import torch
import yaml

from src.pipelines.cnn_trainer import ImageTrainer
from src.pipelines.grid_search import grid_search, cross_validation
from src.utils.constants import DATASET_NAME_MAPPING
from src.utils.model_service import ModelService
from src.pipelines.trainer import Trainer
from torch import cuda

from src.utils.logger import Logger, FoldLogger, CrossValLogger, MultiInstanceLogger

TRAIN_COMMAND: Final[str] = "train"
TRAIN_IMAGE_COMMAND: Final[str] = "train_image"
GRID_SEARCH_COMMAND: Final[str] = "grid_search"
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
        choices=[TRAIN_COMMAND, TRAIN_IMAGE_COMMAND, PREDICT_COMMAND, EVAL_COMMAND, GRID_SEARCH_COMMAND],
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
    # Parse command line arguments
    parser = setup_parser()
    args, _ = parser.parse_known_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Pop first parameter
    n_folds = config["training_parameters"].pop("n_folds")
    run_name = config.pop("run_name")

    # Initialize logger
    logger = MultiInstanceLogger(name=run_name, n_folds=n_folds) if args.pipeline == GRID_SEARCH_COMMAND else Logger(run_name) if n_folds == 0 else CrossValLogger(n_folds, run_name)
    logger.write_config(config)

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
    if model_name.startswith("hetero"):
        model_attributes['cell_types'] = config['dataset_parameters']['cell_types']

    # Create graph dataset
    # Extracting dataset parameters from config
    dataset_parameters = config.pop("dataset_parameters")
    dataset_name = dataset_parameters.pop("class_name")
    process_dataset = dataset_parameters.pop("process")

    # Create instance for dataset and process if given in config
    dataset = DATASET_NAME_MAPPING[dataset_name](**dataset_parameters)

    if process_dataset:
        dataset.process()

    # Log which patients are used in the dataset
    logger.write_text("patient_settings", str(dataset.patient_settings))

    if 'path_image_inputs' in dataset_parameters.keys():
        model_attributes["image_size"] = dataset.image_size
        model_attributes["device"] = device

    if (args.pipeline == TRAIN_COMMAND) & (n_folds == 0):
        model = ModelService.create_model(model_name=model_name, model_attributes=model_attributes)
        logger.write_model(model)

        trainer = Trainer(
            dataset=dataset,
            model=model,
            device=device,
            logger=logger,
            **config.pop("training_parameters"))

        trainer.start_training()
        trainer.save_model()

    if (args.pipeline == TRAIN_COMMAND) & (n_folds > 0):
        cross_validation(model_name=model_name,
                         model_attributes=model_attributes,
                         logger=logger,
                         dataset=dataset,
                         device=device,
                         training_parameters=config['training_parameters'],
                         n_folds=n_folds
                         )

    if args.pipeline == TRAIN_IMAGE_COMMAND:
        model = ModelService.create_model(model_name=model_name, model_attributes=model_attributes)
        logger.write_model(model)

        trainer = ImageTrainer(
            dataset=dataset,
            model=model,
            device=device,
            logger=logger,
            **config.pop("training_parameters"))

        trainer.start_training()
        trainer.save_model()

    if args.pipeline == GRID_SEARCH_COMMAND:
        results = grid_search(model_name=model_name,
                              model_attributes=model_attributes,
                              logger=logger,
                              dataset=dataset,
                              device=device,
                              training_parameters=config['training_parameters'],
                              n_folds=n_folds
                              )
        logger.close()

    if args.pipeline == EVAL_COMMAND:
        model = ModelService.create_model(model_name=model_name, model_attributes=model_attributes)
        logger.write_model(model)

        trainer = Trainer(
            dataset=dataset,
            model=model,
            device=device,
            logger=logger,
            **config.pop("training_parameters"))

        trainer.evaluate()
        trainer.save_model()
    elif args.pipeline == PREDICT_COMMAND:
        pass


if __name__ == "__main__":
    main()
