import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


class DataHandler:
    def __init__(self):
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_data(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()

    def normalize_data(self):
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

    def reshape_data(self):
        self.x_train = self.x_train.reshape((self.x_train.shape[0], 28, 28, 1))
        self.x_test = self.x_test.reshape((self.x_test.shape[0], 28, 28, 1))

    def explore_data(self):
        print('Training data shape:', self.x_train.shape)
        print('Training labels shape:', self.y_train.shape)
        print('Test data shape:', self.x_test.shape)
        print('Test labels shape:', self.y_test.shape)

    def visualize_data(self, num_images=5):
        plt.figure(figsize=(10,5))
        for i in range(num_images):
            plt.subplot(1, num_images, i+1)
            plt.imshow(self.x_train[i].squeeze(), cmap='gray')
            plt.title(self.class_names[self.y_train[i]])
            plt.axis('off')
        plt.show()

class CNNModel:
    def __init__(self, input_shape):
        self.model = None
        self.input_shape = input_shape

    def build_model(self):
        self.model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10)  # 10 classes
        ])

    def compile_model(self):
        self.model.compile(optimizer='adam',
                           loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def summary(self):
        self.model.summary()

class Trainer:
    def __init__(self, model, data_handler):
        self.model = model
        self.data_handler = data_handler
        self.history = None

    def train(self, epochs=10):
        self.history = self.model.fit(
            self.data_handler.x_train,
            self.data_handler.y_train,
            epochs=epochs,
            validation_data=(self.data_handler.x_test, self.data_handler.y_test)
        )

    def plot_training_history(self):
        if self.history is None:
            print("No training history found. Train the model first.")
            return

        # Plot accuracy
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(self.history.history['accuracy'], label='Train')
        plt.plot(self.history.history['val_accuracy'], label='Test')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1,2,2)
        plt.plot(self.history.history['loss'], label='Train')
        plt.plot(self.history.history['val_loss'], label='Test')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

class Evaluator:
    def __init__(self, model, data_handler):
        self.model = model
        self.data_handler = data_handler

    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(
            self.data_handler.x_test,
            self.data_handler.y_test,
            verbose=2
        )
        print('\nTest accuracy:', test_acc)

class Predictor:
    def __init__(self, model, data_handler):
        self.model = model
        self.data_handler = data_handler
        self.probability_model = None
        self.predictions = None

    def prepare_model(self):
        self.probability_model = keras.Sequential([
            self.model,
            layers.Softmax()
        ])

    def predict(self):
        self.predictions = self.probability_model.predict(self.data_handler.x_test)

    def predict_single(self, index):
        if self.predictions is None:
            self.predict()
        predicted_class = np.argmax(self.predictions[index])
        actual_class = self.data_handler.y_test[index]
        print('Predicted class:', self.data_handler.class_names[predicted_class])
        print('Actual class:', self.data_handler.class_names[actual_class])

class Visualizer:
    def __init__(self, predictor, data_handler):
        self.predictor = predictor
        self.data_handler = data_handler

    def plot_image(self, i):
        predictions_array = self.predictor.predictions[i]
        true_label = self.data_handler.y_test[i]
        img = self.data_handler.x_test[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img.squeeze(), cmap='gray')

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% (True: {})".format(
            self.data_handler.class_names[predicted_label],
            100*np.max(predictions_array),
            self.data_handler.class_names[true_label]),
            color=color)

    def plot_predictions(self, num_rows=5, num_cols=3):
        if self.predictor.predictions is None:
            self.predictor.predict()

        num_images = num_rows * num_cols
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, num_cols, i+1)
            self.plot_image(i)
        plt.tight_layout()
        plt.show()

def main():
    # Initialize
    data_handler = DataHandler()
    data_handler.load_data()
    data_handler.explore_data()
    data_handler.visualize_data()
    data_handler.normalize_data()
    data_handler.reshape_data()

    # Build
    cnn_model = CNNModel(input_shape=(28,28,1))
    cnn_model.build_model()
    cnn_model.compile_model()
    cnn_model.summary()

    #Training model
    trainer = Trainer(cnn_model.model, data_handler)
    trainer.train(epochs=10)
    trainer.plot_training_history()

    #Evaluate model
    evaluator = Evaluator(cnn_model.model, data_handler)
    evaluator.evaluate()

    #Predictions
    predictor = Predictor(cnn_model.model, data_handler)
    predictor.prepare_model()
    predictor.predict_single(0)

    #Visualize
    visualizer = Visualizer(predictor, data_handler)
    visualizer.plot_predictions()

if __name__ == "__main__":
    main()
