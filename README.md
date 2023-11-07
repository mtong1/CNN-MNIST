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

// Implementation details would go here

## Decision

After our primary research, we made a deliberate choice to prioritize learning over computational efficiency. Our project focused on the creation of a custom CNN to identify handwritten digits from the MNIST dataset. We chose to include numpy libraries because it includes matrix math and implementation of mathematical procedures. Numpy was a good balance between understanding the theory and math while being computationally fast.

Our initial focus was on forward propagation, where we aimed to understand how input data traverses through the layers of a CNN to produce meaningful predictions. As we advanced in our project, we also implemented backward propagation.

## Challenges

A significant challenge in our computer vision project was scoping. Initially, we struggled to define a clear project goal due to our limited knowledge of machine learning and convolutional neural networks. Our journey started with the goal of implementing a neural network for person identification, then progressed to understanding and integrating forward and backward propagation into TensorFlow, followed by experimenting with the MNIST dataset for both forward and backward propagation. Another major hurdle was comprehending the intricacies of backward propagation, which required extensive hours of rewatching and researching to grasp the underlying mathematical concepts.

## Improvements

If we had more time, our first step would be to improve the efficiency of our code. Next, we would attempt to create a model for identifying people in images. However, during our initial research, we might encounter performance issues due to the lengthy processing time. In response, we would pivot towards using a machine learning library that specializes in convolutional neural networks to expedite our progress. Our focus would shift from understanding the intricacies of the models to their practical application.

## Lessons

Our team learned multiple valuable lessons for future robotic programming projects. Firstly, we recognized the importance of conducting thorough preliminary research before setting specific project goals. This helped us gain a better understanding of the problem and its potential challenges. Additionally, we realized the significance of creating a well-defined roadmap with clear deadlines and goals. The open-ended nature of the project initially posed difficulties in maintaining focus and organization. In the future, having a structured plan will be highly beneficial when tackling more open-ended and complex problems, ensuring that the team stays on track and achieves its objectives effectively.

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
