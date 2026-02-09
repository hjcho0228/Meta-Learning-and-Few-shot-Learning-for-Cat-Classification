🚀 Getting Started
To use the real-time detection system, please follow these steps in order:

1. Prerequisites
Before running the scripts, ensure you have the pre-trained YOLOv5 weights:

Download yolov5s.pt and place it in the project root directory.

2. Training & Evaluation
The system utilizes a ResNet architecture trained via meta-learning. You must prepare the model first:

Train: Execute meta-learning.py to train the ResNet model.

Test: Verify the model's performance by running meta-testing.py.

3. Execution
Once the model is trained and the weights are ready, you can launch the live detection system:

python livestream-detection.py
