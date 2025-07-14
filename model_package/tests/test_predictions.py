from cnn_model.predict import predict_label


def test_predictions_type(sample_input_data):

    predictions = predict_label(sample_input_data[0], 110)

    assert len(predictions) == sample_input_data[1].shape[0]
