import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define new data directory (the path to your data inside the project)
input_dir = r"C:\Users\renad\Desktop\My-Valet-main\My-Valet-main\New WinRAR ZIP archive\clf-data\clf-data"  # Ensure this path points to the correct folder in your project
categories = ['empty', 'not_empty']

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))  # Resize the image to 15x15
        data.append(img.flatten())  # Flatten the image into a 1D array
        labels.append(category_idx)  # Append the label (empty or not_empty)

data = np.asarray(data)
labels = np.asarray(labels)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the classifier (SVM)
classifier = SVC()

# Set up GridSearchCV to find the best model parameters
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

# Train the model using GridSearchCV
grid_search.fit(x_train, y_train)

# Get the best estimator
best_estimator = grid_search.best_estimator_

# Predict using the test set
y_prediction = best_estimator.predict(x_test)

# Calculate model accuracy
score = accuracy_score(y_prediction, y_test)

print(f'{score * 100:.2f}% of samples were correctly classified.')

# Save the trained model
os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)  # Ensure 'models' folder exists
model_path = os.path.join(os.getcwd(), 'models', 'model.pkl')
pickle.dump(best_estimator, open(model_path, 'wb'))
print("The trained model has been saved at:", model_path)
