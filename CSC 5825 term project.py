"""You can find thw ewhole dataset here : https://www.kaggle.com/datasets/gpiosenka/100-bird-species """

import tensorflow as tf
import pandas as pd
import random 
import PIL

"""Data Loading Section"""

#Data Loading

df = pd.read_csv("birds.csv")

df.head()

"""This will split the dataset into different parts, one is the complete set (train_df) and the other only has 200 species in it (train_df_200)"""

random.seed(123)
number_of_clases = 200
class_ids = list(df["class id"].unique())
used_class_ids = random.sample(class_ids, 200)
train_df_200 = df[df['data set'] == "train"]
val_df_200 = df[df['data set'] == "valid"]
test_df_200 = df[df['data set'] == "test"]
df = df[df["class id"].isin(used_class_ids)]
train_df = df[df['data set'] == "train"]
val_df = df[df['data set'] == "valid"]
test_df = df[df['data set'] == "test"]

"""This takes the data from the step above and transforms it into ImageDataGenerators. These are useful because tf.keras.model can directly take the Generator variables and do everything else with it"""

generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

batch_size = 256
seed = 123
target_size = (64,64)

#These will generate images that are full color and 64x64 pixels. To get black and white images change the color_mode to "grayscale"
#To change the image resolution change the target size

train_images = generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepaths',
    y_col='labels',
    color_mode='rgb',
    class_mode='categorical',
    target_size=target_size,
    batch_size=batch_size,
    shuffle=True,
    seed=seed,
    subset='training'
)

val_images = generator.flow_from_dataframe(
    dataframe=val_df,
    x_col='filepaths',
    y_col='labels',
    color_mode='rgb',
    class_mode='categorical',
    target_size=target_size,
    batch_size=batch_size,
    shuffle=True,
    seed=seed,
    subset='validation'
)

test_images = generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepaths',
    y_col='labels',
    color_mode='rgb',
    class_mode='categorical',
    target_size=target_size,
    batch_size=batch_size,
    shuffle=False
)

"""We can look at an individual image in the dataset using the code below"""

import matplotlib.pyplot as plt
#In general to access data in the generators you use the .next() method to retreive an entire batch (256 images), you then need to acces inside the array using [0]
#and then you can access specific images with another [#] where # is the image. Because the dataset is a generator it will give a new image each time you call .next()
#This will be a lower resolution image because we imported it as a 64x64 image when we built the generators, if you want a different resolution adjust the target size in the import step
image = train_images.next()[0][0]
plt.imshow(image.numpy().astype("uint8"))

"""Initial Data exploratory"""

#Create some example charts, how many images do we have, what are the classes sizes like, images sizes, # of channels, etc.

"""Image Augmentation"""

#Augment images by flipping them horizontally, vertically. Rotation. Could even try cropping and padding with black or white pixel
#Not sure if we should only augment the training data or all of the data
def resize_and_rescale(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image = (image / 255.0)
  return image, label

def augment(image_label, seed):
  #From https://www.tensorflow.org/tutorials/images/data_augmentation
  #Don't need to use all of the image augmentation steps for each image, maybe only on half of them?
  #Probably a good idea to make a copy of the image and add that to the data set rather than replacing images
  image, label = image_label
  image, label = resize_and_rescale(image, label)
  image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
  # Make a new seed.
  new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
  # Random crop back to the original size.
  image = tf.image.stateless_random_crop(
      image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
  # Random brightness.
  image = tf.image.stateless_random_brightness(
      image, max_delta=0.5, seed=new_seed)
  # Random Flipping
  image = tf.image.stateless_random_flip_left_right(
      image, seed = new_seed)
  image = tf.image.stateless_random_flip_up_down(
      image, seed = new_seed)
  #random contrast
  image = tf.image.stateless_random_contrast(
      image, lower = 0.5, upper = 1.0, seed = new_seed)
  #random crop
  image = = tf.image.stateless_random_crop(
      image, size=[, , ], seed=seed)
  image = tf.clip_by_value(image, 0, 1)
  return image, label

augmentedDataset =

"""Model Creation"""

#Create the model, want to use a CNN based model. Look at assignment #3 for an example of building a CNN based model
#This is the model I used for asssignemnt 3, we will need to change it but can use it as a starting point
#We should train two models, one using the original data and one using the augmented data
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3), padding = "same", strides = (2), activation="relu", input_shape = (224,224, 1)))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3), padding = "valid", strides = (2)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3), padding = "same", strides = (2), activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3), padding = "valid", strides = (2)))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3), padding = "same", strides = (2), activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3), padding = "valid", strides = (2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, activation = "relu"))
model.add(tf.keras.layers.Dense(200, activation = "softmax"))

model.build()

augmentedDataModel = tf.keras.clone_model(model)

optimizer = tf.optimizers.Adam(learning_rate = 0.0001)
loss_fn = tf.losses.CategoricalCrossentropy()
acc_fn = tf.metrics.CategoricalAccuracy()
model.compile(optimizer, loss_fn, metrics = [acc_fn])
augmentedDataModel.compile(optimizer, loss_fn, metrics = [acc_fn])

#Not sure how to set up a vision transformer so that can be explorerd here:
#There is this for tf https://keras.io/examples/vision/image_classification_with_vision_transformer/
#We can also see if there is a pretrained general transformer out there that we could use as a base
#ViT =

"""Differences between 200 classes, 500 classes, transfer learning

Model Training
"""

#Train the models

model.fit(train_images, batch_size=512, epochs=5, validation_data= val_images)
augmentedDataModel.fit(train_aug_images, batch_size=512, epochs=5, validation_data= val_images)

"""Test the model"""

loss, acc = model.evaluate(test_images)
loss_aug, acc_aug = augmentedDataModel.evaluate(test_images)

"""Show the model Results"""

#Take a random sampling of the test results and show birds that were correctly classified and ones that were incorrectly classified along with the models guess and the right answer

"""Bird Sex classification"""

#This is the "extra" that we can try for if we have time
#Because the dataset doesn't have labels for the sex of the birds, we will have to try some unsupervised learning
#This doesn't have to have good results but we can show that we tried and discuss what we could further do to improve it