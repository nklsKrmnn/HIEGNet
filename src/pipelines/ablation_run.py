from typing import Type

from logger.multi_instance_logger import MultiInstanceLogger
from pipelines.cnn_trainer import ImageTrainer
from pipelines.cross_validation import multi_init_evaluation
from pipelines.trainer import Trainer


def ablation_run(model_name: str,
                 model_attributes: dict,
                 logger: MultiInstanceLogger,
                 dataset,
                 device,
                 trainer_class: Type[Trainer] | Type[ImageTrainer],
                 training_parameters: dict,
                 ablation_parameters: dict,
                 n_test_initialisations: int = 20) -> None:

    # Get ablation variables
    abl_var = list(ablation_parameters.keys())[0]


    for i, abl_value in enumerate(ablation_parameters[abl_var]):
        print("###################################")
        print(f"Ablation {i + 1}/{len(ablation_parameters[abl_var])}")
        print("###################################")
        temp_logger = logger.next_logger()

        if abl_var in model_attributes.keys():
            if (model_attributes[abl_var] != 'abl') and i == 0:
                raise ValueError(f"Variable {abl_var} is not set to 'abl' in model attributes")
            model_attributes[abl_var] = abl_value
        if abl_var in training_parameters.keys():
            if (training_parameters[abl_var] != 'abl') and i == 0:
                raise ValueError(f"Variable {abl_var} is not set to 'abl' in training parameters")
            training_parameters[abl_var] = abl_value
        else:
            raise ValueError(f"Variable {abl_var} not found in model or training parameters")

        temp_logger.write_text("config_model_parameters", {model_name:model_attributes})
        temp_logger.write_text("config_training_parameters", training_parameters)

        multi_init_evaluation(model_name=model_name,
                              model_attributes=model_attributes,
                              logger=temp_logger,
                              dataset=dataset,
                              device=device,
                              trainer_class=trainer_class,
                              training_parameters=training_parameters,
                              n_test_initialisations=n_test_initialisations)
