# AI Task for Fashion-MNIST Clothing Classification

## Difine Problem

The Fashion-MNIST dataset contain 60,000 small square 28×28 pixel grayscale images of items and there is 10 clases.

* T-shirt/top
* Trouser
* Pullover
* Dress
* Coat
* Sandal
* Shirt
* Sneaker
* Bag
* Ankle boot

So, it is a multi-class classification problem that whay I used convolutional neural networks.

## Import Libraries
I import all the necessary libraries for this problem.
## Load Dataset
Load dataset from Fashion-MNIST.

## Data Analysis
I check the shape of training and test data, then explore the grayscale image data.

## Data Preprocessing
Reshape the data for scaling and encode the target variable by one hot encoding. For preparing pixel data I convert data from integer to float then normalize data to range 0-1.

## Define Model
I used a convolutional layer with a small filter size (3,3) and a modest number of filters (32) followed by a max pooling layer in the problem-part1.py file. Then, I increased filters and padding to improve model performance. Because it is a multiclass classification problem I used a softmax activation function. Between the feature extractor and the output layer, I added a dense layer to interpret the features, in this case with 100 nodes. All layers used the ReLU activation function. I used a conservative configuration for the stochastic gradient descent optimizer with a learning rate of 0.01 and a momentum of 0.9.

## Evaluate Model
After the model was defined, I evaluated the model by using 5-fold cross-validation and set the test set is 20% of the training dataset. I trained the model for a modest 10 training epochs with a default batch size of 32 examples. The model achieved 91+ accuracy. Finally, the performances by curves, box, and whisker plots are created to summarize the distribution of accuracy scores.

## For Evaluation
I solved this problem in the Kaggle notebook and then implemented those codes in my local computer to maintain the evaluation procedure. First I create a virtual environment in my local computer then install necessary dependencies in the virtual environment and create a requirements.txt file. I stuck in the "evaluate_model.py" file, I just solved the problem in Kaggle. New experience here for this type of task, but I know, I'm capable, I just need guidance.


# AI-Task
