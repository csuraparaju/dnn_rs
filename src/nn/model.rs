use nalgebra::{DMatrix};
use crate::nn::layers::Linear;
use crate::nn::loss::MSE;
use crate::nn::activation::ReLU;

/**
    * We can think of a neural network (NN) as a mathematical function
    * which takes an input data x and computes an output y: y = fNN (x)
    * The function fNN is a nested function, where each sub function
    * can be thought of as a layer in the neural network. The output of
    * one layer is passed as input to the next layer, and so on, until
    * we reach the final layer which produces the output y. That is,
    * y = fNN (x) = fL (fL-1 ( ... f2 (f1 (x)) ... )) where f1, f2, ... fL
    * are vector valued functions of the form fi (x) = g_i(W_i * x + b_i).
    * Here, W_i is the weight matrix, b_i is the bias vector, and g_i is the
    * activation function applied element-wise to the input. The parameters
    * W_i and b_i are learned during the training process using an optimization
    * algorithm such as gradient descent.
    *
    * The NeuralNetwork struct is a high-level abstraction that encapsulates
    * the entire neural network model. It consists of a sequence of layers
    * and a loss function. Currently, only a sequential model is supported,
    * where each layer is a linear layer followed by an activation function.
**/

pub struct NeuralNetwork {
    pub layers: Vec<Box<Linear>>,
    pub loss: Box<MSE>,
    pub activations: Vec<Box<ReLU>>
}

impl NeuralNetwork {
    // Constructor for the NeuralNetwork struct. Creates a new neural network
    // model with the specified layers and loss function.
    pub fn new(layers: Vec<Box<Linear>>, loss: Box<MSE>, activations: Vec<Box<ReLU>>) -> Self {
        NeuralNetwork {
            layers: layers,
            loss: loss,
            activations: activations
        }
    }

    // During forward propagation, we pass the input data x through each layer
    // in the neural network to compute the output y. That is, y = fNN (x) = fL (fL-1 ( ... f2 (f1 (x)) ... ))
    pub fn forward(&mut self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut A = x.clone();
        for i in 0..self.layers.len() {
            A = self.layers[i].forward(&A);
            A = self.activations[i].forward(&A);
        }
        return A;
    }

    // During backward propagation, we compute the gradients of the loss with
    // respect to the parameters of the neural network. Given the loss function
    // L, we can compute the gradients of the loss with respect to the output
    // of the neural network A, and then backpropagate these gradients through
    // each layer to compute the gradients of the loss with respect to the
    // parameters of the neural network.
    pub fn backward(&mut self) {
        let dLdA = self.loss.backward();
        for i in (0..self.layers.len()).rev() {
            let dLdZ = self.activations[i].backward(&dLdA);
            self.layers[i].backward(&dLdZ);
        }
    }
}
