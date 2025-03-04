import torch
from typing import Type

from src.logger.cross_val_logger import ManyFoldLogger
from src.pipelines.cnn_trainer import ImageTrainer
from src.pipelines.trainer import Trainer
from src.utils.model_service import ModelService

#TODO docstrings

def cross_validation(model_name: str,
                     model_attributes: dict,
                     logger: ManyFoldLogger,
                     dataset,
                     device,
                     trainer_class,
                     training_parameters: dict,
                     n_folds: int = 5
                     ) -> None:
    """

    :param model_name:
    :param model_attributes:
    :param logger:
    :param dataset:
    :param device:
    :param training_parameters:
    :param n_folds:
    :return: None
    """
    dataset.create_folds(n_folds)

    for fold in range(n_folds):
        model = ModelService.create_model(model_name=model_name,
                                          model_attributes=model_attributes)
        logger.fold_logger[fold].write_model(model)
        dataset.activate_fold(fold)

        trainer = trainer_class(
            dataset=dataset,
            model=model,
            device=device,
            logger=logger.fold_logger[fold],
            **training_parameters)

        trainer.start_training()
        trainer.save_model()

    logger.summarize()
    logger.close()


def multi_init_evaluation(model_name: str,
                          model_attributes: dict,
                          logger: ManyFoldLogger,
                          dataset,
                          device,
                          trainer_class: Type[Trainer] | Type[ImageTrainer],
                          training_parameters: dict,
                          n_test_initialisations: int = 10
                          ) -> None:
    """

    :param trainer_class:
    :param model_name:
    :param model_attributes:
    :param logger:
    :param dataset:
    :param device:
    :param training_parameters:
    :param n_test_initialisations:
    :return: None
    """

    for fold in range(n_test_initialisations):
        print('##################################')
        print(f'Test initialisation {fold}/{n_test_initialisations}')
        print('##################################')

        #TODO: Make a nicer way to load models
        model = ModelService.create_model(model_name=model_name,
                                          model_attributes=model_attributes,
                                          fold=fold)
        logger.fold_logger[fold].write_model(model)

        if training_parameters['epochs'] > 0 and 'model_dir' in model_attributes.keys():
            raise Warning('Model is loaded but epochs are not set to 0. Training will be performed.')

        trainer = trainer_class(
            dataset=dataset,
            model=model,
            device=device,
            logger=logger.fold_logger[fold],
            **training_parameters)

        trainer.start_training()
        if training_parameters['epochs'] > 0:
            trainer.load_best_model(model_name=model_name,
                                    model_attributes=model_attributes)
        trainer.evaluate()

        # Increasing torch seed by one
        torch.manual_seed(torch.initial_seed() + fold + 1)

    logger.summarize()
    logger.save_test_scores("./data/output/test_scores.csv")
    logger.close()
