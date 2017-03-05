# Notes

# Dataset Summary
I used numpy's array's shape property to get the dataset size.

# Exploratory Visualizationz
I've created an image of sample signs. Every row has 10 samples of the sames sign.
In order to to this, I've created a dictionary with image label keys and lists of image samples as elements.
After this, i've simply concatenadted the elements of the map into a final sample image.

# Preprocessing
I haven't made much preprocessing; the only thing is to converting the image pixel values from integers to floats.
(this is because the model gets arrays of float32's as an input)

# Model Architecture
I have 2 convolutional layer (a 4x4 and a 3x3 kernel). Both convolutional layers have a relu activation element, and reducing the element number with a 2x2 maxpool layer.

The output of the second convolutional layer is flattened out to a 1440 x 1 layer.

After this a first fully connected layer converts it into a 500x1 layer; and a second fully connected layer converts into a 43-element layer (the number of labels). This layer is an output, and supposed to return the probability of each sign for the given input image.

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

In every batch I'm printing out the cost for the given batch, the accuracy for the given batch and the accuracy for the validatin class.

I've noticed that the accuracy for a given training batch is always higher than for the validation model. It makes sense, as it means that the un-known validation set has some features which are not included in the training set.

I've also made a shuffle in the training set, because the signs are ordered by label; so without shuffling every batch would have a lot of similar signs.


# Acquiring New Image
I've got the sample images from the dataset itself.


# Performance on New Images
I've made a performance test both on validation data and on test data; the model performes similarly.

# Model Certainty - Softmax Probabilities
When analyzing the probabilities it shows that that 2nd and 3rd most 


