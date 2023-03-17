import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from torch import nn
import sys


def setup_model(model_path: str):
    # A transform that normalise the input to the model similar to how it was trained
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # FieldDay model architecture
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(labels))
    model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))

    # load your model file here
    model.load_state_dict(torch.load(model_path))
    model.eval()

    def predict(input):
        input = transform(input)
        input = input.to(torch.float)
        input = torch.unsqueeze(input, 0)
        input = input.to("cpu")
        output = model(input)
        idx = torch.argmax(output).item()
        return idx

    return predict


def run_webcam(predict, labels):
    # NOTE: If this index doesn't work, try 0
    cam = cv2.VideoCapture(1)
    cv2.startWindowThread()
    cv2.namedWindow("Webcam")
    # This is the loop that takes the webcam feed and runs it through the model
    while True:
        ret, frame = cam.read()
        frame = frame[:, :, [2, 1, 0]]
        frame = Image.fromarray(frame)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        # flip the frame like most webcam feeds
        frame = cv2.flip(frame, 1)

        # eval frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input = Image.fromarray(frame_rgb)
        idx = predict(input)

        center = frame.shape
        w = center[1] / 2
        h = center[0]

        x = center[1] / 2 - w / 2
        y = center[0] / 2 - h / 2

        cropped = frame[int(y) : int(y + h), int(x) : int(x + w)]

        print("Press Escape to close webcam window.")
        print("Predicted Class: ", labels[idx])
        print("\033c", end="")

        cv2.imshow("Webcam", cropped)

        k = cv2.waitKey(30)
        if k % 256 == 27:
            # ESC pressed
            # K in multiples of 27 means the escape key was pressed
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    with open("labels.txt", "r") as f:
        labels = [s.strip() for s in f.readlines()]

    model_path = sys.argv[1]
    predict = setup_model(model_path)
    run_webcam(predict, labels)
