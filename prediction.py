#!/usr/bin/env python

from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np

class_labels = [
    "Negative",
    "Positive"
]

# Load the json file that contains the model's structure
f = Path("PotholeDataset/models/model_structure.json")
model_structure = f.read_text()

model = model_from_json(model_structure)
model.load_weights("PotholeDataset/models/model_weights.h5")

img = image.load_img("positive.jpg", target_size=(150, 150))
image_to_test = image.img_to_array(img) #numpy array

# needed because keras expects a 4D array but we are only testing 1 image
list_of_images = np.expand_dims(image_to_test, axis=0)

results = model.predict(list_of_images) # array that holds prediction for each image passed

# check the image result
single_result = results[0]

# Figure out which class has the highest score
most_likely_class_index = int(np.argmax(single_result)) # get highest result
class_likelihood = single_result[most_likely_class_index]
class_label = class_labels[most_likely_class_index] # get the class label

# Print the result
print("This is image is {} with Likelihood: {:2f}".format(class_label, class_likelihood))
