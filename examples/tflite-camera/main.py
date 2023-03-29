import time
import cv2
import numpy as np
import sys
from tensorflow import lite
from tensorflow.keras.layers import Normalization


def setup_model(model_path: str):
    # Load model from path and allocate tensors
    interpreter = lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get details of input and output tensor
    input_tensor_index = interpreter.get_input_details()[0]['index']
    output_tensor_index = interpreter.get_output_details()[0]['index']

    # Prepares a function that will make a prediction for an input image
    def predict(input):
        interpreter.set_tensor(input_tensor_index, input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_tensor_index)[0]
        return output

    return predict


def setup_preprocessing(cap):
    ret, frame = cap.read()

    # Creates a `normalize` function to normalize the input similarly to the training
    means = [.485, .456, .406]
    variances = [np.square(.229), np.square(.224), np.square(.225)]
    normalize = Normalization(mean=means, variance=variances)

    # Determines how to crop the input image to get a square, center-cropped image
    center = frame.shape
    width = center[1]
    height = center[0]
    x = 0
    y = 0

    if width > height:
        size = height
        x = (width - height) / 2
    else:
        size = width
        y = (height - width) / 2

    # Prepares a function that will preprocess an input image for prediction
    def preprocess(frame):
        frame = np.array(frame)
        center_cropped = frame[int(y): int(y + size), int(x): int(x + size)]
        resized = cv2.resize(center_cropped, (224, 224))
        normalized = normalize(np.divide(resized, 255))
        transposed = np.transpose(normalized, (2, 0, 1))
        input = transposed[np.newaxis, :]
        return input

    return preprocess


def setup_presentation(labels):
    # Prepares a function that will present labels and confidence based on the model prediction
    def present(output):
        results = dict(zip(labels, output))
        sorted_results = sorted(
            results.items(),
            key=lambda item: item[1],
            reverse=True
        )
        for item in sorted_results:
            print(item[1], '\t', item[0])
        return sorted_results[0][0]
    return present


def run_camera(predict, present, cam):
    # This is the loop that takes the webcam feed and runs it through the model
    while True:
        ret, frame = cam.read()

        start_time = time.time()

        input = preprocess(frame)
        output = predict(input)
        prediction = present(output)

        frame_time = time.time() - start_time

        print("\nPredicted Class: ", prediction)
        print("Running at:", int(1/frame_time), "FPS")
        print("Press Escape to exit.")
        print("\033c", end="")

        cv2.imshow("TFLite Camera", frame)

        k = cv2.waitKey(30)
        if k % 256 == 27:
            # ESC pressed
            # K in multiples of 27 means the escape key was pressed
            break


if __name__ == "__main__":
    model_path = sys.argv[1]

    with open("labels.txt", "r") as f:
        labels = [s.strip() for s in f.readlines()]

    # NOTE: If this index doesn't work, try 0
    cam = cv2.VideoCapture(1)
    cv2.startWindowThread()
    cv2.namedWindow("TFLite Camera")

    preprocess = setup_preprocessing(cam)
    predict = setup_model(model_path)
    present = setup_presentation(labels)

    run_camera(predict, present, cam)

    cam.release()
    cv2.destroyAllWindows()
