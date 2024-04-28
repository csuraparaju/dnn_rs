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
    pub activations: Vec<Box<ReLU>>, // Invariant: activations.len() == layers.len()
    pub loss: Box<MSE>,
}

impl NeuralNetwork {
    // Constructor for the NeuralNetwork struct. Creates a new NeuralNetwork
    // model with the specified layers and loss function.
    pub fn new(layers: Vec<Box<Linear>>, activations: Vec<Box<ReLU>>, loss: Box<MSE>) -> Self {
        NeuralNetwork {
            layers: layers,
            activations: activations,
            loss: loss,
        }
    }


    // During forward propagation, we apply a sequence of linear transformations
    // and activation functions to the input data x to obtain the output data y.
    // That is, y = fNN (x) = fL (fL-1 ( ... f2 (f1 (x)) ... )). The forward
    // method computes the output of the neural network given the input data x.
    pub fn forward(&mut self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut A = x.clone();
        for i in 0..self.layers.len() {
            A = self.layers[i].forward(&A);
            if i > self.activations.len() - 1 {
                break;
            }
            A = self.activations[i].forward(&A);
        }
        return A;
    }

    // During backward propagation, we compute the gradients of the loss with
    // respect to the parameters of the neural network. Given the gradients
    // of the loss with respect to the output of the neural network, we can
    // compute the gradients of the loss with respect to the parameters of
    // each layer in the neural network. The backward method computes the
    // gradients of the loss with respect to the parameters of the neural
    // network using the chain rule of calculus.
    pub fn backward(&mut self) {
        let mut dLdA = self.loss.backward();
        let mut dLdZ = DMatrix::zeros(0,0);
        for i in (0..self.layers.len()).rev() {
            if i <= self.activations.len() - 1 {
                dLdZ = self.activations[i].backward(&dLdA);
            }
            dLdA = self.layers[i].backward(&dLdZ);
        }
    }
}
