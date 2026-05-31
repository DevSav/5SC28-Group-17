def evaluate_model(model, test_inputs, test_outputs):
    """Evaluate a model with mean squared error."""
    predictions = model.predict(test_inputs)

    if predictions is None:
        print("No predictions yet.")
        return None

    error = test_outputs - predictions
    mse = (error ** 2).mean()
    return mse

