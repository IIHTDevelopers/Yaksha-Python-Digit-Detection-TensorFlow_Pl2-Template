import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load and preprocess dataset
def load_and_preprocess():
    """
    TODO:
    - Load the MNIST dataset using tf.keras.datasets.mnist.load_data().
    - Flatten the 28x28 images into vectors of length 784.
    - Normalize the pixel values to the range [0, 1].
    - Use train_test_split to select 10,000 samples from X and y for training.
    - Return X_train, X_test, y_train, y_test
    """
    # Example return statement
    # return X_train, X_test, y_train, y_test
    pass


# Step 2: Build the neural network model
def build_model():
    """
    TODO:
    - Use tf.keras.Sequential to create a model with:
        - Dense(128, activation='relu', input_shape=(784,))
        - Dense(64, activation='relu')
        - Dense(10, activation='softmax') for classification (digits 0–9)
    - Compile the model using:
        - optimizer = 'adam'
        - loss = 'sparse_categorical_crossentropy'
        - metrics = ['accuracy']
    - Return the compiled model
    """
    # Example return statement
    # return model
    pass


# Step 3: Train the model
def train_model(model, X_train, y_train):
    """
    TODO:
    - Train the model using model.fit() for:
        - epochs = 10
        - batch_size = 32
        - verbose = 1
    - Return the trained model
    """
    # Example return statement
    # return model
    pass


# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    TODO:
    - Evaluate the trained model on X_test and y_test.
    - Use model.evaluate() to get loss and accuracy.
    - Print the accuracy using:
        print(f"Test Accuracy: {accuracy:.4f}")
    - Return accuracy as float (optional, for testing)
    """
    # Example return statement
    # return accuracy
    pass


# Step 5: Predict a sample digit
def predict_sample(model):
    """
    TODO:
    - Load the first sample from the MNIST test set.
    - Reshape and normalize it just like training data.
    - Use model.predict() to get class probabilities.
    - Use np.argmax() to get the predicted digit.
    - Print the predicted digit.
    - Return the predicted digit (int)
    """
    # Example return statement
    # return predicted_class
    pass


# Step 6: Main function to run all steps
def main():
    """
    TODO:
    - Load and preprocess the data
    - Build the model
    - Train the model
    - Evaluate it on test data
    - Predict one sample
    - No need to return anything
    """
    # Example flow:
    # X_train, X_test, y_train, y_test = load_and_preprocess()
    # model = build_model()
    # model = train_model(model, X_train, y_train)
    # evaluate_model(model, X_test, y_test)
    # predict_sample(model)
    pass


# Entry point
if __name__ == "__main__":
    main()
