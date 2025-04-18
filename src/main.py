import argparse
from typing import Final

import torch
from torch import cuda
import yaml

from pipelines.ablation_run import ablation_run
from src.pipelines.cnn_trainer import ImageTrainer
from src.pipelines.grid_search import grid_search
from src.pipelines.cross_validation import cross_validation, multi_init_evaluation
from src.utils.constants import DATASET_NAME_MAPPING
from src.utils.model_service import ModelService
from src.pipelines.trainer import Trainer
from src.logger.logger import Logger
from src.logger.multi_instance_logger import MultiInstanceLogger
from src.logger.cross_val_logger import ManyFoldLogger
from src.utils.validate_config import validate_config

TRAIN_COMMAND: Final[str] = "train"
TRAIN_IMAGE_COMMAND: Final[str] = "train_image"
GRID_SEARCH_COMMAND: Final[str] = "grid_search"
EVAL_COMMAND: Final[str] = "evaluate"
PREDICT_COMMAND: Final[str] = "predict"
ABLATION_COMMAND: Final[str] = "ablation"

GRAPH_COMMAND: Final[str] = "graph"
IMAGE_COMMAND: Final[str] = "image"


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
        "--trainer",
        "-t",
        required=True,
        choices=[GRAPH_COMMAND, IMAGE_COMMAND],
        help="Which trainer class to use.",
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        required=True,
        choices=[TRAIN_COMMAND, TRAIN_IMAGE_COMMAND, PREDICT_COMMAND, EVAL_COMMAND, GRID_SEARCH_COMMAND, ABLATION_COMMAND],
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

    # Check validity of the config file values if not image_trainer
    if args.trainer == 'graph':
        validate_config(config)

    # Pop first parameter
    n_folds = config["training_parameters"].pop("n_folds")
    n_test_init = config["training_parameters"].pop("n_test_initialisations") if "n_test_initialisations" in config[
        "training_parameters"] else 1
    run_name = config.pop("run_name")

    # Initialize logger
    n_logger = n_test_init if args.pipeline in [ABLATION_COMMAND, EVAL_COMMAND] else n_folds
    n_folds = n_test_init if args.pipeline == ABLATION_COMMAND else n_folds
    if args.pipeline == GRID_SEARCH_COMMAND or args.pipeline == ABLATION_COMMAND:
        logger = MultiInstanceLogger(name=run_name, n_folds=n_folds)
    elif n_logger <= 1:
        logger = Logger(run_name)
    else:
        logger = ManyFoldLogger(n_logger, run_name)
    logger.write_dict(config)

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
            print(f"[MAIN]: Random seed set to {seed} for both CPU and GPU.")
        else:
            print(f"[MAIN]: Random seed set to {seed} for CPU.")

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
    # logger.write_text("patient_settings", str(dataset.patient_settings))

    if args.trainer == 'graph':
        trainer_class = Trainer
    elif args.trainer == 'image':
        trainer_class = ImageTrainer
    else:
        raise ValueError(f"Trainer class '{args.trainer}' is not supported.")

    if dataset.image_size is not None:
        model_attributes["image_size"] = dataset.image_size
        model_attributes["device"] = device

    if (args.pipeline == TRAIN_COMMAND) & (n_folds <= 1):
        model = ModelService.create_model(model_name=model_name, model_attributes=model_attributes)
        logger.write_model(model)

        trainer = trainer_class(
            dataset=dataset,
            model=model,
            device=device,
            logger=logger,
            **config.pop("training_parameters"))

        trainer.start_training()
        trainer.save_model()

    if (args.pipeline == TRAIN_COMMAND) & (n_folds > 1):
        cross_validation(model_name=model_name,
                         model_attributes=model_attributes,
                         logger=logger,
                         dataset=dataset,
                         device=device,
                         trainer_class=trainer_class,
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
        if 'gs_start_index' not in config.keys():
            config['gs_start_index'] = 1
        results = grid_search(model_name=model_name,
                              model_attributes=model_attributes,
                              logger=logger,
                              dataset=dataset,
                              device=device,
                              trainer_class=trainer_class,
                              training_parameters=config['training_parameters'],
                              n_folds=n_folds,
                              start_index=config['gs_start_index'] - 1
                              )
        logger.close()

    if (args.pipeline == EVAL_COMMAND) and (n_test_init <= 1):

        if 'test_patients' in dataset_parameters.keys():
            if (dataset_parameters['test_split'] == 0.0) or (dataset_parameters['test_patients'] == []) or (config['training_parameters']['reported_set'] == []):
                print("\033[31m!!! WARNING !!!\033[0m")
                print("\033[31mWARNING: Evaluation requested. Some parameter my is uncommon.\033[0m")
                print("\033[31mWARNING: Recommended: test_split > 0.0, test_patients != [], reported_set=='test'.\033[0m")

        run_training = 'model_name' not in model_attributes.keys()

        model = ModelService.create_model(model_name=model_name, model_attributes=model_attributes.copy())
        logger.write_model(model)

        trainer = trainer_class(
            dataset=dataset,
            model=model,
            device=device,
            logger=logger,
            **config["training_parameters"])

        # Train model if model is not provided
        if run_training:
            trainer.start_training()
            trainer.save_model()

        trainer.evaluate()

    if (args.pipeline == EVAL_COMMAND) and (n_test_init > 1):
        multi_init_evaluation(model_name=model_name,
                              model_attributes=model_attributes,
                              logger=logger,
                              dataset=dataset,
                              device=device,
                              trainer_class=trainer_class,
                              training_parameters=config['training_parameters'],
                              n_test_initialisations=n_test_init
                              )

    if args.pipeline == ABLATION_COMMAND:
        ablation_run(model_name=model_name,
                     model_attributes=model_attributes,
                     logger=logger,
                     dataset=dataset,
                     device=device,
                     trainer_class=trainer_class,
                     training_parameters=config['training_parameters'],
                     ablation_parameters=config['ablation_parameters'],
                     n_test_initialisations=n_test_init)

    elif args.pipeline == PREDICT_COMMAND:
        pass


if __name__ == "__main__":
    main()
