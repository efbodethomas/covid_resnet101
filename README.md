# Deep Learning Image Classification with Transfer Learning

This repository contains a deep learning code implementation for image classification using convolutional neural networks (CNNs) with transfer learning. The code focuses on classifying chest X-ray images as either normal or COVID-19 using a dataset.

## Dataset
The dataset used in this project consists of two classes: normal and COVID-19. The chest X-ray images for each class were collected and preprocessed. The dataset was split into training and testing sets to evaluate the model's performance.

## Code Overview
The code is implemented in Python using TensorFlow and Keras libraries. Here's a summary of the major components and their functionalities:

1. Loading and Preprocessing: The script loads the chest X-ray images from the dataset, resizes them to a standard size (224x224 pixels), and normalizes the pixel values.

2. Data Split: The loaded data is split into training and testing sets using the `train_test_split` function from scikit-learn. This ensures that the model is trained on a subset of the data and evaluated on unseen data.

3. Model Architecture: The base model used for transfer learning is ResNet50, pre-trained on the ImageNet dataset. It is imported from `tensorflow.keras.applications` and integrated into a custom CNN architecture. The model also includes data augmentation layers for improved generalization.

4. Model Training: The model is compiled with the Adam optimizer and binary cross-entropy loss. The training is performed with early stopping to prevent overfitting. The training progress is visualized using accuracy and loss plots.

5. Model Evaluation: After training, the model is evaluated on the test set using the `evaluate` method. The test loss and accuracy are reported. Predictions are made on the test set, and a confusion matrix and classification report are generated to assess the model's performance.

6. Model Saving: The trained model is saved using the `save` method provided by Keras. This allows for easy reusability and deployment of the model for future tasks.

## Usage
To run the code, follow these steps:

1. Install the required dependencies specified in the `requirements.txt` file.
2. Prepare your dataset and adjust the file paths in the code accordingly.
3. Execute the code in a Python environment with TensorFlow and Keras installed.

## Results
The trained model achieves a certain level of accuracy and demonstrates its effectiveness in classifying chest X-ray images as normal or COVID-19. The results are presented in the form of accuracy and loss plots, a confusion matrix, and a classification report.

## Conclusion
This code provides an example implementation of image classification using deep learning and transfer learning. By leveraging a pre-trained model like ResNet101, the model can benefit from the knowledge gained during training on the ImageNet dataset. This approach improves the accuracy and generalization capabilities of the model, making it suitable for various image classification tasks.
