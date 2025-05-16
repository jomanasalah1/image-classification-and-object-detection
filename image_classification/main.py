import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import warnings
import mlflow
import mlflow.keras
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Or your MLflow server URI
mlflow.set_experiment("Image_Classification1")

# Loading dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Data preprocessing
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Convert labels to categorical
categories = 10
train_y = to_categorical(train_labels, categories)
test_y = to_categorical(test_labels, categories)

# Labels for CIFAR-10
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

with mlflow.start_run():
    # Model architecture
    classifier = Sequential(name='CIFAR10')
    bit_depth = 3  # for color images

    # Convolutional layers
    classifier.add(Convolution2D(32, (3, 3), input_shape=(32, 32, bit_depth), activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(32, (3, 3), activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Convolution2D(64, (3, 3), activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(64, (3, 3), activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Convolution2D(128, (3, 3), activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(128, (3, 3), activation='relu'))
    classifier.add(Dropout(0.5))

    # Fully connected layers
    classifier.add(Flatten())
    classifier.add(Dense(units=64, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=32, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=16, activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=10, activation='softmax'))

    # Compile the model
    opt = SGD(learning_rate=0.01)
    classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Log model parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("epochs", 100)
    
    # Define callback to save best model
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    # Train the model
    history = classifier.fit(
        train_images, 
        train_y,
        batch_size=64,
        epochs=100,
        validation_data=(test_images, test_y),
        callbacks=[checkpoint]
    )

    # Log metrics
    for epoch in range(len(history.history['accuracy'])):
        mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

    # Load the best saved model
    best_model = load_model('best_model.h5')
    
    # Evaluate the best model
    loss, accuracy = best_model.evaluate(test_images, test_y, verbose=0)
    print(f'\nBest Model Test Accuracy: {accuracy*100:.2f}%')
    print(f'Best Model Test Loss: {loss:.4f}')
    
    # Log evaluation metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_loss", loss)

    # Make predictions
    pred = best_model.predict(test_images)
    pred_labels = np.argmax(pred, axis=1)
    test_labels_flat = test_labels.ravel()
    
    # Log classification report
    clf_report = classification_report(test_labels_flat, pred_labels, target_names=labels, output_dict=True)
    mlflow.log_dict(clf_report, "classification_report.json")
    
    # Log model
    mlflow.keras.log_model(
        best_model,
        "model",
        registered_model_name="CIFAR10_CNN_Classifier"
    )
    
    # Save and log artifacts
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'r', label='Training Loss')
    plt.plot(history.history['val_loss'], 'g', label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], 'r', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], 'g', label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    mlflow.log_artifact("training_history.png")
    
    # Sample predictions visualization
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.ravel()
    for i in np.arange(0, 25):
        axes[i].imshow(test_images[i])
        axes[i].set_title(f"Actual: {labels[test_labels[i][0]]}\nPredicted: {labels[pred_labels[i]]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()
    mlflow.log_artifact("predictions.png")

    # Create and save results
    actual = [labels[label] for label in test_labels_flat]
    predictions = [labels[label] for label in pred_labels]
    results = pd.DataFrame({'Actual': actual, 'Predictions': predictions})
    results.to_csv('classification_results.csv', index=False)
    mlflow.log_artifact("classification_results.csv")