import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix


# Φόρτωση MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Κανονικοποίηση στο [0,1]
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

# Προσαρμογή διαστάσεων για το δίκτυο
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

def create_target_labels(images, labels):
    """
    Δημιουργεί έναν πίνακα στόχων όπου κάθε δείγμα αντιστοιχεί σε δείγμα της επόμενης κυκλικής ετικέτας.
    :param images: Πίνακας εικόνων (numpy array)
    :param labels: Πίνακας ετικετών (numpy array)
    :return: Πίνακας με εικόνες-στόχους (numpy array), Πίνακας με ετικέτες-στόχους (numpy array)
    """
    unique_labels = np.unique(labels)
    target_images = np.zeros_like(images)
    target_labels = np.zeros_like(labels)

    for label in unique_labels:
        # Υπολογισμός της επόμενης ετικέτας
        next_label = (label + 1) % 10

        # Εύρεση δειγμάτων με την τρέχουσα και την επόμενη ετικέτα
        current_indices = np.where(labels == label)[0]
        next_indices = np.where(labels == next_label)[0]

        if len(next_indices) == 0:
            raise ValueError(f"Δεν υπάρχουν δείγματα για την κλάση {next_label}.")

        # Εξασφάλιση επαναληπτικής χρήσης δειγμάτων από την target κλάση
        repeated_next_indices = np.tile(next_indices, (len(current_indices) // len(next_indices) + 1))[:len(current_indices)]

        # Αντιστοίχιση 1-1
        for i, current_index in enumerate(current_indices):
            target_images[current_index] = images[repeated_next_indices[i]]
            target_labels[current_index] = next_label  # Το νέο label είναι το next_label

    return target_images, target_labels



y_train_images, y_train_labels = create_target_labels(x_train, y_train)
y_test_images, y_test_labels = create_target_labels(x_test, y_test)


# Ορισμός latent space
latent_dim = 64

# Encoder
input_img = keras.Input(shape=(28, 28, 1))
x = keras.layers.Flatten()(input_img)
x = keras.layers.Dense(latent_dim, activation="relu")(x)
encoder = keras.Model(input_img, x, name="encoder")

# Decoder
decoder_input = keras.Input(shape=(latent_dim,))
x = keras.layers.Dense(28 * 28, activation="sigmoid")(decoder_input)
x = keras.layers.Reshape((28, 28, 1))(x)
decoder = keras.Model(decoder_input, x, name="decoder")

# Autoencoder (χρήση των encoder & decoder)
autoencoder = keras.Model(inputs=input_img, outputs=decoder(encoder(input_img)))

# Compile & Train
autoencoder.compile(optimizer="adam", loss="mse")
start_time = time.time()
history = autoencoder.fit(x_train, y_train_images, epochs=20, batch_size=256, validation_data=(x_test, y_test_images))


# Δοκιμή ανακατασκευής εικόνων
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# Επιλέγουμε 3 τυχαίους δείκτες από τα δεδομένα
n = 3  # Αριθμός δειγμάτων
indices = np.random.choice(len(x_test), n, replace=False)  # Τυχαίοι δείκτες

plt.figure(figsize=(10, 4))
for i, idx in enumerate(indices):
    # Πρωτότυπες εικόνες
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
    ax.axis("off")

    # Αποκωδικοποιημένες εικόνες
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[idx].reshape(28, 28), cmap="gray")
    ax.axis("off")

plt.show()

# Απεικόνιση της απώλειας
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss during Encoder Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Ελεγχός autoencoder σε εικόνες με θόρυβο

noise_factor = 0.7
x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)


x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Εκπαίδευση με θορυβώδεις εικόνες αλλά καθαρές ως στόχο
autoencoder.fit(x_train_noisy, y_train_images, epochs=20, batch_size=256, validation_data=(x_test, y_test_images))

# Δοκιμή ανακατασκευής εικόνων
encoded_imgs_noisy = encoder.predict(x_test)
decoded_imgs_noisy = decoder.predict(encoded_imgs_noisy)

# Επιλέγουμε 3 τυχαίους δείκτες από τα δεδομένα
n = 3  # Αριθμός δειγμάτων
indices = np.random.choice(len(x_test), n, replace=False)  # Τυχαίοι δείκτες

plt.figure(figsize=(10, 4))
for i, idx in enumerate(indices):
    # Πρωτότυπες εικόνες
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
    ax.axis("off")

    # Αποκωδικοποιημένες εικόνες
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_noisy[idx].reshape(28, 28), cmap="gray")
    ax.axis("off")

plt.show()


#Φτιάχνω τον Classifier

inputs = keras.Input(shape=(28,28))
x = keras.layers.Flatten()(inputs)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs= outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(decoded_imgs, y_test_labels)
print('Test accuracy:', test_acc)


# Κάνουμε προβλέψεις στο test set
predictions = model.predict(decoded_imgs)
predicted_labels = np.argmax(predictions, axis=1)

# Εντοπισμός σωστών και λανθασμένων προβλέψεων
correct_indices = np.where(predicted_labels == y_test_labels)[0]
incorrect_indices = np.where(predicted_labels != y_test_labels)[0]

# Επιλογή 4 τυχαίων σωστών και 4 τυχαίων λανθασμένων προβλέψεων
n_samples = 4  # Αριθμός περιπτώσεων
correct_samples = np.random.choice(correct_indices, n_samples, replace=False)
incorrect_samples = np.random.choice(incorrect_indices, n_samples, replace=False)

# Οπτικοποίηση των σωστών προβλέψεων
plt.figure(figsize=(8, 8))
plt.suptitle("Correct Predictions")
for i, index in enumerate(correct_samples):
    ax = plt.subplot(2, 2, i + 1)
    plt.imshow(decoded_imgs[index].reshape(28, 28), cmap="gray")
    plt.title(f"True: {y_test_labels[index]}, Pred: {predicted_labels[index]}")
    ax.axis("off")

plt.show()

# Οπτικοποίηση των λανθασμένων προβλέψεων
plt.figure(figsize=(8, 8))
plt.suptitle("Incorrect Predictions")
for i, index in enumerate(incorrect_samples):
    ax = plt.subplot(2, 2, i + 1)
    plt.imshow(decoded_imgs[index].reshape(28, 28), cmap="gray")
    plt.title(f"True: {y_test_labels[index]}, Pred: {predicted_labels[index]}")
    ax.axis("off")

plt.show()

# Υπολογισμός του πίνακα σύγχυσης
conf_matrix = confusion_matrix(y_test_labels, predicted_labels)

# Οπτικοποίηση του πίνακα σύγχυσης χωρίς Seaborn
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Προσθήκη ετικετών στους άξονες
tick_marks = np.arange(10)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)

# Προσθήκη αριθμών μέσα στα κελιά του πίνακα
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

test_loss, test_acc_noisy = model.evaluate(decoded_imgs_noisy, y_test_labels)
print('Test accuracy- Noisy Training set:', test_acc_noisy)

# Κάνουμε προβλέψεις στο test set
predictions = model.predict(decoded_imgs_noisy)
predicted_labels = np.argmax(predictions, axis=1)

# Εντοπισμός σωστών και λανθασμένων προβλέψεων
correct_indices = np.where(predicted_labels == y_test_labels)[0]
incorrect_indices = np.where(predicted_labels != y_test_labels)[0]

# Επιλογή 4 τυχαίων σωστών και 4 τυχαίων λανθασμένων προβλέψεων
n_samples = 4  # Αριθμός περιπτώσεων
correct_samples = np.random.choice(correct_indices, n_samples, replace=False)
incorrect_samples = np.random.choice(incorrect_indices, n_samples, replace=False)

# Υπολογισμός του πίνακα σύγχυσης
conf_matrix = confusion_matrix(y_test_labels, predicted_labels)

# Οπτικοποίηση του πίνακα σύγχυσης χωρίς Seaborn
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Προσθήκη ετικετών στους άξονες
tick_marks = np.arange(10)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)

# Προσθήκη αριθμών μέσα στα κελιά του πίνακα
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

