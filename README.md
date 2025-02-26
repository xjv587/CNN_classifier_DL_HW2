# Convolutional Neural Network (CNN) Classifier with Logging and Model Training

This project focuses on implementing and training a Convolutional Neural Network (CNN) classifier in PyTorch to classify images into six categories. The model is designed to output class logits in a (B,6) tensor format. The project includes logging, training, and evaluation functionalities, making it a well-rounded deep learning implementation.

# Key Components:

- CNN Implementation:
  - Developed a CNNClassifier using torch.nn.Conv2d and other layers.
  - Ensured the model outputs logits for six classes.

- Training and Evaluation:
  - Trained the CNN model and saved it as cnn.th.
  - Achieved a test accuracy of 89% to meet project goals.
  - Prevented overfitting by carefully handling dataset normalization.

- Performance Monitoring with TensorBoard:
  - Logged training loss at every iteration using torch.utils.tensorboard.SummaryWriter.
  - Recorded training and validation accuracy at each epoch to track progress.
  - Visualized model predictions to gain insights into performance.
