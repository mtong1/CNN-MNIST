# Convolutional Neural Network for Person Detection and Following

**Authors**: An Grocki, Anmol Sandhu, Madie Tong  
**Course**: Computational Introduction to Robotics Fall 2023

## Goal

Our goal for this project was to delve deeply into the mathematics, logic, and theoretical foundations behind Convolutional Neural Networks (CNNs). We recognized that gaining a comprehensive understanding of neural networks required more than just theoretical knowledge, so our primary objective was to implement a CNN from the ground up. We avoided the use of common machine learning libraries such as TensorFlow and Keras because they obscured the intricate details of CNNs and had multiple unknown functions.

We chose to focus on CNNs because they are important and relevant in computer vision and robotics.

## Solution

### Math Behind the CNN

What differentiates a CNN versus other neural networks is with its use of convolutions/correlations. To understand this fundamentally, we used various resources and took notes. All mathematical explanations for the processes used in our implementation can be found in this [pdf](./CNN-math.pdf).

### Implementation

The CNN-MNIST project is structured as a series of Python modules that collectively build a convolutional neural network (CNN) designed to classify handwritten digits from the MNIST dataset. Below is an overview of the implementation details for each component of the project:

#### Network Initialization

- The `NN` class represents the neural network and is composed of a list of layers.
- The network is initialized with a sequence of layers that include convolutional layers, activation layers (Sigmoid), a reshape layer, and dense layers.
- The `Convolution` class is responsible for the convolution operations within the network, initialized with the input shape, kernel size, and depth.
- The `Dense` class represents fully connected layers, initialized with input and output sizes.
- The `Sigmoid` class implements the sigmoid activation function.
- The `Reshape` class is used to reshape the input data between layers when necessary.

#### Data Preprocessing

- The MNIST dataset is loaded from an HDF5 file (`MNISTdata.hdf5`) and preprocessed using the `preprocess_data` function within `CNN.py`.
- The data is reshaped and normalized as required by the network.

#### Training Loop (`CNN.py`)

- The `NN` class's `train` method is called to start the training process, which iterates over the dataset for a specified number of epochs.
- During each epoch, a forward pass predicts the output, and the loss is computed using the mean squared error (MSE) function defined in `losses.py`.
- A backward pass is performed to update the weights and biases of the network layers using the gradients calculated from the loss.

#### Forward and Backward Propagation

- Each layer class (`Convolution`, `Dense`, `Sigmoid`, `Reshape`) implements a `forward` method to propagate the input data through the network and a `backward` method to propagate the error gradient during training.
- The `Convolution` layer performs the convolution operation using the `correlate2d` function from the `correlate` module.
- The `Dense` layer performs matrix multiplication with the input data and adds a bias term.
- The `Sigmoid` layer applies the sigmoid activation function to the input data.
- The `Reshape` layer reshapes the input data to match the expected input shape of the next layer.

### Testing and Evaluation

- After training, the network's performance is evaluated on the test dataset.
- The accuracy is calculated by comparing the network's predictions to the true labels of the test data.

#### Main Function (`CNN.py`)

- The main function serves as the entry point for the script, where the network is created, trained, and evaluated.
- It initializes the network with the defined layers and calls the training function with the preprocessed data.

The modular design of the project allows for easy modification and extension of the network architecture. Each layer operates independently and can be adjusted or replaced as needed to experiment with different network configurations.

## Decision

After our primary research, we made a deliberate choice to prioritize learning over computational efficiency. Our project focused on the creation of a custom CNN to identify handwritten digits from the MNIST dataset. We chose to include numpy libraries because it includes matrix math and implementation of mathematical procedures. Numpy was a good balance between understanding the theory and math while being computationally fast.

Our initial focus was on forward propagation, where we aimed to understand how input data traverses through the layers of a CNN to produce meaningful predictions. As we advanced in our project, we also implemented backward propagation.

## Challenges

A significant challenge in our computer vision project was scoping. Initially, we struggled to define a clear project goal due to our limited knowledge of machine learning and convolutional neural networks. Our journey started with the goal of implementing a neural network for person identification, then progressed to understanding and integrating forward and backward propagation into TensorFlow, followed by experimenting with the MNIST dataset for both forward and backward propagation. Another major hurdle was comprehending the intricacies of backward propagation, which required extensive hours of rewatching and researching to grasp the underlying mathematical concepts.

## Improvements

If we had more time, our first step would be to improve the efficiency of our code. Next, we would attempt to create a model for identifying people in images. However, during our initial research, we might encounter performance issues due to the lengthy processing time. In response, we would pivot towards using a machine learning library that specializes in convolutional neural networks to expedite our progress. Our focus would shift from understanding the intricacies of the models to their practical application.

## Lessons

Our team learned multiple valuable lessons for future robotic programming projects. Firstly, we recognized the importance of conducting thorough preliminary research before setting specific project goals. This helped us gain a better understanding of the problem and its potential challenges. Additionally, we realized the significance of creating a well-defined roadmap with clear deadlines and goals. The open-ended nature of the project initially posed difficulties in maintaining focus and organization. In the future, having a structured plan will be highly beneficial when tackling more open-ended and complex problems, ensuring that the team stays on track and achieves its objectives effectively.

### Potential Improvements for Future Work

- **Increase Convolution Layers**: The current network has only one convolution layer. Adding more convolutional layers can help the network learn more complex features at various levels of abstraction.
- **Add Pooling Layers**: After convolution layers, pooling layers can be used to reduce the dimensionality of the feature maps, which helps in reducing the number of parameters and computational complexity.
- **Normalization Layers**: We can also consider adding batch normalization layers after convolution layers or fully connected layers to stabilize and speed up the network's training.
- **Activation Functions**:
  - **ReLU Activation**: ReLU and its variants (like Leaky ReLU or Parametric ReLU) often perform better than Sigmoid in deep networks because they help alleviate the vanishing gradient problem.
- **Loss Functions**:
  - **Categorical Cross-Entropy**: When working with multi-class classification, as with MNIST, categorical cross-entropy is often more suitable than mean squared error (MSE) for the loss function.

## Resources

- [Convolutional Neural Network from Scratch | Mathematics & Python Code](https://www.youtube.com/watch?v=Lakz2MoHy6o)
- [But what is a neural network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3)
- [What is backpropagation really doing? | Chapter 3, Deep learning](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4)
- [Backpropagation calculus | Chapter 4, Deep learning](https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4)
- [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA)
- [The Quest of Higher Accuracy for CNN Models | by Swanand Mhalagi | Towards Data Science](https://towardsdatascience.com/the-quest-of-higher-accuracy-for-cnn-models-42df5d731faf#:~:text=Increase%20model%20capacity%20,or%20to%20even%20higher%20size)
- [neural networks - What is the effect of using pooling layers in CNNs? - Artificial Intelligence Stack Exchange](https://ai.stackexchange.com/questions/21532/what-is-the-effect-of-using-pooling-layers-in-cnns#:~:text=Pooling%20has%20multiple%20benefits%20Robust,Basically%20pooling)
- [How ReLU and Dropout Layers Work in CNNs | Baeldung on Computer Science](https://www.baeldung.com/cs/ml-relu-dropout-layers#:~:text=Computing%20the%20ReLU%20This%20function,is%20respectively%20negative%20or%20not)
- [machine learning - Why is cross entropy loss better than MSE for multi-class classification? - Cross Validated](https://stats.stackexchange.com/questions/573944/why-is-cross-entropy-loss-better-than-mse-for-multi-class-classification#:~:text=81%201%203%20I%27m%20not,not%20on%20the%20same%20scale)
