# Cat-Dogs-classification using Convolutional Neural Networks (CNNs)


This project aims to classify images of animals using Convolutional Neural Networks (CNNs). The model is trained on a dataset containing images of 20000 different cats and dogs.

## Getting Started 

To run the code on your local machine, follow these steps:

1. Open a new google colab notebook.
2. Obtain the `kaggle.json` file from the Kaggle website under your account settings (you have to sign up on kaggle).
3. Upload the `kaggle.json` file in the root directory of the project on colab notebook.
4. Connect your gpu backend in the colab notebook.
5. Run the code given in the repo in the same order.

## Usage

Once the code is executed, the model will be trained on the provided dataset. After training, you can upload any image of an animal dogg/cat to the Colab notebook and use the provided snippet (test_img = cv2.imread('/content/doggy.jpeg')) to test the model's classification accuracy.

## How It Works

This project utilizes concepts from Convolutional Neural Networks (CNNs) to classify images. The model architecture is based on the VGG16 architecture, a popular CNN architecture known for its effectiveness in image classification tasks.

### Challenges Faced

While developing this project, several challenges were encountered:

Certainly! Here's a concise and professional version of the challenges section:

## Challenges Faced

1. **Accuracy vs. Overfitting Trade-off**: Balancing high accuracy on training data with overfitting was challenging, especially with limited datasets and complex CNN architectures. Techniques like regularization and batch normalization were crucial for mitigating overfitting.

2. **Dataset Quality and Quantity**: Ensuring dataset quality and quantity significantly impacted model performance. Datasets with fewer images or containing bad data posed challenges, necessitating innovative solutions such as data augmentation and dataset splitting.

3. **Multi-Class Classification Techniques**: Transitioning to multi-class classification required implementing techniques like "categorical" encoding and one-hot encoding. Understanding softmax and sigmoid activation functions was essential for effective multi-class classification.

4. **Integration of Pre-trained Models**: Integrating pre-trained models into the architecture introduced challenges, addressed using tools like Gemini for seamless integration. Leveraging pre-trained models enhanced model performance and efficiency.

5. **Evaluation Strategies**: Exploring evaluation strategies such as K-Fold cross-validation and early stopping techniques facilitated robust model assessment, addressing issues like overfitting and ensuring generalization to unseen data.

6. **Padding and Scaling**: Understanding the impact of padding options and scaling input data was crucial. Making informed decisions regarding these parameters improved model convergence and performance.

## Acknowledgments

- PreGrad: Lectures on AI and Machine Learning concepts.
- TensorFlow documentation: Resources for understanding CNN architectures and techniques.
- Kaggle: Provided datasets used for training and testing the model.
- Gemini from Google.                        

