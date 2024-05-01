use nalgebra::DMatrix;
use crate::nn::layers::Layer;
use crate::nn::loss::LossFunction;
use crate::nn::activation::ActivationFunction;

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

pub trait NeuralNetwork {
    fn forward(&mut self, x : &DMatrix<f64>) -> DMatrix<f64>;
    fn backward(&mut self) -> ();
    fn get_layers(&self) -> Vec<Box<dyn Layer>>;
    fn get_activations(&self) -> Vec<Box<dyn ActivationFunction>>;
    fn get_loss(&self) -> Box<dyn LossFunction>;
}
pub struct SequentialNeuralNetwork {
    pub layers: Vec<Box<dyn Layer>>,
    pub activations: Vec<Box<dyn ActivationFunction>>,
    pub loss: Box<dyn LossFunction>,
}

impl SequentialNeuralNetwork {
    // Constructor for the NeuralNetwork struct. Creates a new NeuralNetwork
    // model with the specified layers and loss function.
    pub fn new(layers: Vec<Box<dyn Layer>>,
               activations: Vec<Box<dyn ActivationFunction>>,
               loss: Box<dyn LossFunction>) -> Self {
    SequentialNeuralNetwork {
            layers: layers,
            activations: activations,
            loss: loss,
        }
    }
}

impl NeuralNetwork for SequentialNeuralNetwork {


    // During forward propagation, we apply a sequence of linear transformations
    // and activation functions to the input data x to obtain the output data y.
    // That is, y = fNN (x) = fL (fL-1 ( ... f2 (f1 (x)) ... )). The forward
    // method computes the output of the neural network given the input data x.
    fn forward(&mut self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut A = x.clone();
        for i in 0..self.layers.len() {
            A = self.layers[i].forward(&A);
            // If there is no activation function, we return the output
            // of the previous layer as the network's output
            if i > self.activations.len() - 1 {
                return A;
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
    fn backward(&mut self) -> () {
        let mut dLdA = self.loss.backward();
        let mut dLdZ = DMatrix::zeros(0,0);
        for i in (0..self.layers.len()).rev() {
            if i <= self.activations.len() - 1 {
                dLdZ = self.activations[i].backward(&dLdA);
            }
            dLdA = self.layers[i].backward(&dLdZ);
        }
    }

    fn get_layers(&self) -> Vec<Box<dyn Layer>>{
        self.layers.clone()
    }

    fn get_activations(&self) -> Vec<Box<dyn ActivationFunction>>{
        self.activations.clone()
    }

    fn get_loss(&self) -> Box<dyn LossFunction>{
        self.loss.clone()
    }
}
