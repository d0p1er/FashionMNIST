# Visualization of the internal state of a neural network based on dataset FashionMNIST
> sorry for my English

# Content
1. [Introduction](#intro)
2. [Information bottleneck](#ib)
3. [Hard negatives](#hn)
4. [Confusion matrix](#conf)


<a name="intro"></a>
## 1. Introduction
Fashion-MNIST is a dataset of Zalando's article images—consisting of
a training set of 60,000 examples and a test set of 10,000 examples.
Each example is a 28x28 grayscale image, associated with a label from 10 classes. 

1. T-Shirt
2. Trousers
3. Pullover
4. Dress 
5. Coat 
6. Sandals 
7. Shirt 
8. Sneakers 
9. Bag 
10. AnkleBoots


<a name="ib"></a>
## 2. Information bottleneck
There are 100 samples of 10 of each class.
The result was a short trajectory of the “movement” of the
prediction of each of their samples.

![ib.png](images/train/information_bottleneck.gif)


<a name="hn"></a>
## 3. Hard negatives
There are 5 samples of each class with the maximum softmax
confidence of the erroneous class.

| Correct class |                               Miss 0                               |                               Miss 1                               |                               Miss 2                               |                               Miss 3                               |                               Miss 4                               |
|:-------------:|:------------------------------------------------------------------:|:------------------------------------------------------------------:|:------------------------------------------------------------------:|:------------------------------------------------------------------:|:------------------------------------------------------------------:|
|    T-Shirt    |         ![Shirt](images/classes/T-Shirt/Shirt_0.png) Shirt         |         ![Shirt](images/classes/T-Shirt/Shirt_1.png) Shirt         |         ![Shirt](images/classes/T-Shirt/Shirt_2.png) Shirt         |            ![Bag](images/classes/T-Shirt/Bag_3.png) Bag            |            ![Bag](images/classes/T-Shirt/Bag_4.png) Bag            |
|   Trousers    |          ![Coat](images/classes/Trousers/Coat_0.png) Coat          |    ![Sneakers](images/classes/Trousers/Sneakers_1.png) Sneakers    |    ![Sneakers](images/classes/Trousers/Sneakers_2.png) Sneakers    |          ![Coat](images/classes/Trousers/Coat_3.png) Coat          |    ![Sneakers](images/classes/Trousers/Sneakers_4.png) Sneakers    |
|   Pullover    |          ![Coat](images/classes/Pullover/Coat_0.png) Coat          | ![AnkleBoots](images/classes/Pullover/AnkleBoots_1.png) AnkleBoots |          ![Coat](images/classes/Pullover/Coat_2.png) Coat          |          ![Coat](images/classes/Pullover/Coat_3.png) Coat          |        ![Shirt](images/classes/Pullover/Shirt_4.png) Shirt         |
|     Dress     |             ![Bag](images/classes/Dress/Bag_0.png) Bag             |             ![Bag](images/classes/Dress/Bag_1.png) Bag             |           ![Coat](images/classes/Dress/Coat_2.png) Coat            |          ![Shirt](images/classes/Dress/Shirt_3.png) Shirt          |          ![Shirt](images/classes/Dress/Shirt_4.png) Shirt          |
|     Coat      |          ![Dress](images/classes/Coat/Dress_0.png) Dress           |          ![Dress](images/classes/Coat/Dress_1.png) Dress           |          ![Shirt](images/classes/Coat/Shirt_2.png) Shirt           |          ![Dress](images/classes/Coat/Dress_3.png) Dress           |          ![Shirt](images/classes/Coat/Shirt_4.png) Shirt           |
|    Sandals    | ![AnkleBoots](images/classes/Sandals/AnkleBoots_0.png) AnkleBoots  | ![AnkleBoots](images/classes/Sandals/AnkleBoots_1.png) AnkleBoots  | ![AnkleBoots](images/classes/Sandals/AnkleBoots_2.png) AnkleBoots  | ![AnkleBoots](images/classes/Sandals/AnkleBoots_3.png) AnkleBoots  | ![AnkleBoots](images/classes/Sandals/AnkleBoots_4.png) AnkleBoots  |
|     Shirt     |       ![T-shirt](images/classes/Shirt/T-Shirt_0.png) T-Shirt       |          ![Dress](images/classes/Shirt/Dress_1.png) Dress          |           ![Coat](images/classes/Shirt/Coat_2.png) Coat            |           ![Coat](images/classes/Shirt/Coat_3.png) Coat            |       ![T-Shirt](images/classes/Shirt/T-Shirt_4.png) T-Shirt       |
|   Sneakers    | ![AnkleBoots](images/classes/Sneakers/AnkleBoots_0.png) AnkleBoots | ![AnkleBoots](images/classes/Sneakers/AnkleBoots_1.png) AnkleBoots | ![AnkleBoots](images/classes/Sneakers/AnkleBoots_2.png) AnkleBoots | ![AnkleBoots](images/classes/Sneakers/AnkleBoots_3.png) AnkleBoots | ![AnkleBoots](images/classes/Sneakers/AnkleBoots_4.png) AnkleBoots |
|      Bag      |        ![Sandals](images/classes/Bag/Sandals_0.png) Sandals        |           ![Dress](images/classes/Bag/Dress_1.png) Dress           |        ![T-Shirt](images/classes/Bag/T-Shirt_2.png) T-Shirt        |           ![Dress](images/classes/Bag/Dress_3.png) Dress           |           ![Dress](images/classes/Bag/Dress_4.png) Dress           |
|  AnkleBoots   |   ![Sneakers](images/classes/AnkleBoots/Sneakers_0.png) Sneakers   |   ![Sneakers](images/classes/AnkleBoots/Sneakers_1.png) Sneakers   |   ![Trousers](images/classes/AnkleBoots/Trousers_2.png) Trousers   |   ![Sneakers](images/classes/AnkleBoots/Sneakers_3.png) Sneakers   |   ![Sneakers](images/classes/AnkleBoots/Sneakers_4.png) Sneakers   |


<a name="conf"></a>
## 4. Confusion matrix

![confusion_matrix.png](images/ConfusionMatrix.png) 

