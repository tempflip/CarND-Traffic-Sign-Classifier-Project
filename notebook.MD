# Notes

# Dataset Summary
I used numpy's array's shape property to get the dataset size.

# Exploratory Visualizationz
I've created an image of sample signs. Every row has 10 samples of the same sign.
In order to to this, I've created a dictionary with image label keys and lists of image samples as elements.
After this, i've simply concatenadted the elements of the map into a final sample image.

# Preprocessing
I haven't made much preprocessing; the only thing is converting the image pixel values from integers to floats.
(this is because the model gets arrays of float32's as an input)

# Model Architecture
I have 2 convolutional layers (with 4x4 and a 3x3 kernels). Both convolutional layers have a relu activation element, and reducing the element number with a 2x2 maxpool layer.

The output of the second convolutional layer is flattened out into a 1440 x 1 layer.

After this a first fully connected layer converts it into a 500x1 layer; and a second fully connected layer converts into a 43-element layer (the number of sign labels). This layer is an output, and supposed to return the probability of each sign for the given input image.

# Model Training
I'm using the tf.train.AdamOptimizer optimizer in order to reduce the loss, which is computed as a cross-entrpoy between the output of the model and the real label of the given training sample.


```
logits = model(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y, name="entropy")
cost = tf.reduce_mean(cross_entropy, name="cost")
optimizer = tf.train.AdamOptimizer(learning_rate = rate, name="optimizer").minimize(cost)
```
labels -> the real prediction
logits -> output from the model



# Solution Design
I'm running a training session.  In every epoch I'm running over the training samples, and feeding into the model in batches.

I've noticed that relatively smaller batches are working better, i.e. in large batches the model doesn't learn quickly.

In every batch I'm printing out the cost for the given batch, the accuracy for the given batch and the accuracy for the validation set.

I've noticed that the accuracy for a given training batch is always higher than for the validation set. It makes sense, as it means that the un-known validation set has some features which are not included in the training set.

I've also made a shuffle in the training set, because the signs are ordered by label; so without shuffling every batch would have a lot of similar signs.


# Acquiring New Images
I've got new images from the web. The challange with these images is that they are not neccessarily normalized the same way as the training images, so it is not guaranteed that the sign is on the center of the image, and the color scale might be different. That's why the model performes significantly poorer.


# Performance on New Images
I've made a performance test both on validation data and on test data; the model performes similarly. However, on the new images there are very bad results; I think this is happening because the dataset was normalized.

# Model Certainty - Softmax Probabilities
When analyzing the probabilities it shows that that 2nd and 3rd most likely predictions are usually similar to the first one (for example, speed limit signs are ususally confused with other speed limit signs). This makes sense, as these signs are similar.

I'm NOT using a softmax probabilities when printing out the probabilities, because the results are very high, so a softmax function would reset out every field (and kept only the one with the highest probability). Instead, I'm simply shoving the probabilies from the logit model.
It shows that the probabilities are very close to each other, so the model is pretty uncertain.

```
1    Go straight or right 38%, Go straight or left 34%, Keep left 26%, 
2    Vehicles over 3.5 metric tons prohibited 38%, Children crossing 30%, Road narrows on the right 30%, 
3    End of speed limit (80km/h) 36%, Bicycles crossing 32%, Priority road 31%, 
```





