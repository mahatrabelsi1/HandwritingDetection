Handwriting Recognition with Neural Networks

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits using the MNIST dataset. The model is trained on the MNIST dataset and can predict the digits from both test data and custom uploaded images.

Overview :

        This project uses the popular MNIST dataset to train a CNN model that recognizes handwritten digits (0-9). The model is trained using TensorFlow and Keras, and it can be used to predict digits from both test data and user-uploaded images.

Technologies :

        Python 3.x

        TensorFlow 2.x

        Keras

        Matplotlib

        NumPy

        PIL (Python Imaging Library)


# Define the path to your custom image :
        img_path = 'path_to_your_uploaded_image.png'

# Predict the class for the uploaded image :

        predicted_class = predict_uploaded_image(img_path)

        print(f"Model prediction: {predicted_class}")

File Structure: 

├── HandwritingDetection.ipynb    # Jupyter Notebook with the main code

├── README.md                    
