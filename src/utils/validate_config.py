
def validate_config(config_dict: dict) -> None:

    dataset_params = config_dict["dataset_parameters"]
    training_params = config_dict["training_parameters"]

    # Check if validation patient is given if validation split is given
    if (dataset_params['validation_split'] > 0.0):
        if 'validation_patients' not in dataset_params.keys():
            raise ValueError("No validation patients provided, but validation split is given.")
        if dataset_params['validation_patients'] is None:
            raise ValueError("No validation patients provided, but validation split is given.")

    # Warn if validation patients are given but validation split is not
    if 'validation_patients' in dataset_params.keys() and dataset_params['validation_patients'] is not None:
        if dataset_params['validation_split'] == 0.0:
            print("\033[31mWARNING: Validation patients are provided, but validation split is not given.\033[0m")

    # Check if test patient is given if test split is given
    if (dataset_params['test_split'] > 0.0):
        if 'test_patients' not in dataset_params.keys():
            raise ValueError("No test patients provided, but test split is given.")
        if dataset_params['test_patients'] is None:
            raise ValueError("No test patients provided, but test split is given.")

    # Warn if test patients are given but test split is not
    if 'test_patients' in dataset_params.keys() and dataset_params['test_patients'] is not None:
        if dataset_params['test_split'] == 0.0:
            print("\033[31mWARNING: Test patients are provided, but test split is not given.\033[0m")

    # Warn if test set is reported but validation set is give
    if training_params['reported_set'] == "test" and dataset_params['validation_split'] > 0.0:
        print("\033[31mWARNING: Test set is reported, but validation set is given.\033[0m")

    # Warn that patience is set for a test run
    if training_params['patience'] < training_params['epochs'] and training_params['reported_set'] == "test":
        print("\033[31mWARNING: Patience is set for a test run.\033[0m")


    # Check if targets are one-hot encoded if cross entropy loss is used
    #ce = training_params["loss"] == "crossentropy"
    #oh = dataset_params["onehot_targets"]
    #if (ce and not oh) or (not ce and oh):
    #    raise ValueError("Targets must be one-hot encoded for cross entropy loss and vice versa.")