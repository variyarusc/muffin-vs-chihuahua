# Advait Variyar CAIS++ Winter Project S26

**Name:** Advait Variyar

**USC Email:** variyar@usc.edu

**Project Description:** This project presents a computer vision model to perform binary image classification using Convolutional Neural Networks in PyTorch. The model itself classifies an image as either a 'muffin' or a 'chihuahua'.

## Dataset

The dataset [(link)](https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification/data?select=test) consists of approximately 6000 images of muffins and chihuahuas (balanced). The dataset is already split into seperate train and test sets at a ratio of 80:20.

The raw images in the dataset are of different sizes. Thus, before initializing the train and test DataLoader objects, all images are transformed to  be (224 x 224) pixels. Furthermore, the range of RGB values is transformed from [0, 255] to [-1.0, 1.0]. This allows for faster convergence and better performance while training the model.

## Model Development and Training

For this project, I chose to implement a Convolutional Neural Network due to its advantages for computer vision. Through the process of convolution and pooling, CNNs are able to efficiently identify features within an image.

For my model, I used 3 convolutional layers, setting the kernel size = (3 x 3) and stride = 2. Furthermore, after each convolution, the feature map was pooled using Max Pooling with kernel size = (2 x 2) and stride = 2. Thus, by the end of convolution and pooling, the output was downsampled significantly.

Setting stride = 2 for the convolutional layers helped with regularization towards the end of the project. I found that when stride = 1, the difference in accuracy between classes was >10%. However, after setting stride = 2, this difference in accuracy fell to around 3%, with minimal drop in overall model accuracy.

## Model Evaluation/Results

To evaluate my model, I chose to focus on the accuracy metric. This task does not require special consideration towards false positives or false negatives. Thus, reducing the total amount of incorrect predictions (accuracy) is an appropriate goal.

After allowing the model to run for 10 epochs with batch sizes of 32 samples, the final test accuracy was 72%. The model was able to classify chihuahuas with an accuracy of 73.6% and classified muffins with an accuracy of 70.2%.

## Discussion

#### a) How well does your dataset, model architecture, training procedures, and chosen metrics fit the task at hand? 

The dataset consists of approximately 6000 images of chihuahuas and muffins that are roughly balanced by class. While the dataset was already split into train and test splits, a more thorough approach would be to randomize splits, along with adding a validation set.

Using a Convolutional Neural Network is appropriate for this task as outlined earlier. While some of the parameters were optimized for faster training times, model architecture could be improved by increasing the number of channels that are returned by each convolutional layer.

Finally, accuracy is the appropriate metric to consider for this task as outlined earlier. Metrics such as precision and recall are better suited for tasks where either false positives or false negatives are unacceptable (e.g. Medical Diagnosis, Insurance Claims, Spam Detection). This task has no such factors, so accuracy is still appropriate.

#### b) Can your efforts be extended to wider implications, or contribute to social good? Are there any limitations in your methods that should be considered before doing so?

While this project is trivial at its current stage, computer vision models to differentiate between two similar looking objects can have many important use cases. For instance, correctly identifying animals (like chihuahuas) could be important in computer vision models trying to address issues like animal trafficking or animal abuse.

In order to be applied for such tasks, the model would need a far more extensive dataset consisting of task-specific images (e.g. CCTV footage, broadcast footage). The model would be optimized for recall instead of accuracy in order to reduce false negative predictions.

#### c) If you were to continue this project, what would be your next steps?

1. Increase model depth by adding more convolutional layers
2. Look into advanced techniques like Transfer Learning to improve model performance
3. Train the model on bigger datasets with more ambigious images
4. Use techniques like grid search to optimize hyperparameters
5. Start using validation set for more accurate test metrics