use nalgebra::DMatrix;
use crate::nn::model::{NeuralNetwork, SequentialNeuralNetwork};


/**
    * Stochastic Gradient Descent (SGD) Optimizer
    *
    * The Stochastic Gradient Descent (SGD) optimizer is a simple yet effective
    * optimization algorithm used to update the parameters of a neural network
    * during the training process. The basic idea behind SGD is to update the
    * parameters in the direction of the negative gradient of the loss function
    * with respect to the parameters. This is done by computing the gradient of
    * the loss with respect to the parameters for each sample in the training
    * data, and then updating the parameters using the average gradient over
    * the entire training data.

**/


pub struct SGD {
    pub model: SequentialNeuralNetwork,
    pub lr: f64, // Learning Rate
    pub mu: f64, // Momentum
    pub v_W: Vec<DMatrix<f64>>, // Velocity for weights
    pub v_b: Vec<DMatrix<f64>> // Velocity for biases
}

impl SGD {
    // Constructor for the SGD struct. Creates a new SGD optimizer with
    // the specified learning rate and momentum.
    pub fn new(model: SequentialNeuralNetwork, lr: f64, mu: f64) -> Self {
        let mut v_W: Vec<nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>>> = Vec::new();
        let mut v_b = Vec::new();
        for i in 0..model.layers.len() {
            v_W.push(DMatrix::zeros(model.layers[i].get_weights().nrows(), model.layers[i].get_weights().ncols()));
            v_b.push(DMatrix::zeros(model.layers[i].get_bias().nrows(), model.layers[i].get_bias().ncols()));
        }
        SGD {
            model: model,
            lr: lr,
            mu: mu,
            v_W: v_W,
            v_b: v_b
        }
    }

    // The update method is used to update the parameters of the neural network
    // using the Stochastic Gradient Descent (SGD) algorithm. The update is done
    // by computing the gradient of the loss with respect to the parameters for
    // each sample in the training data, and then updating the parameters using
    // the average gradient over the entire training data.
    pub fn update(&mut self, x: &DMatrix<f64>, y: &DMatrix<f64>) {

        // Forward pass (compute loss)
        let Z = self.model.forward(&x);
        let loss = self.model.loss.forward(&Z, &y);

        // Backward pass (compute gradients)
        let _dLdA = self.model.backward();

        for i in 0..self.model.layers.len() {

            if self.mu == 0.0 {
                // Update the weights and biases using the negative gradient
                // of the loss with respect to the parameters
                let dLdW = self.model.layers[i].get_weight_gradient().clone();
                let dLdb = self.model.layers[i].get_bias_gradient().clone();
                let curr_weights = self.model.layers[i].get_weights();
                let new_weights = curr_weights - self.lr * &dLdW;
                self.model.layers[i].set_weights(new_weights);
                
                let curr_bias = self.model.layers[i].get_bias();
                let new_bias = curr_bias - self.lr * &dLdb;
                self.model.layers[i].set_bias(new_bias);

            } else {
                // Update the weights and biases using momentum
                let dLdW = self.model.layers[i].get_weight_gradient().clone();
                let dLdb = self.model.layers[i].get_bias_gradient().clone();
                self.v_W[i] = self.mu * &self.v_W[i] + &dLdW;
                self.v_b[i] = self.mu * &self.v_b[i] + &dLdb;
                let curr_weights = self.model.layers[i].get_weights();
                let new_weights = curr_weights - self.lr * &self.v_W[i];
                self.model.layers[i].set_weights(new_weights);
                
                let curr_bias = self.model.layers[i].get_bias();
                let new_bias = curr_bias - self.lr * &self.v_b[i];
                self.model.layers[i].set_bias(new_bias);
            }
        }
    }
}