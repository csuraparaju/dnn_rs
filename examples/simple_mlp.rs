use dnn_rs::nn::model::{SequentialNeuralNetwork, NeuralNetwork};
use dnn_rs::nn::activation::{ReLU, Sigmoid, ActivationFunction};
use dnn_rs::nn::layers::{Linear, Layer};
use dnn_rs::nn::loss::{CrossEntropy, MSE};

use dnn_rs::optim::sgd::SGD;

use nalgebra::{DMatrix};

fn main() {
    // Create a simple neural network model
    let linear1 = Linear::new(2, 4);
    let activation1 = Sigmoid::new();
    let linear2 = Linear::new(4, 2);
    let activation2 = Sigmoid::new();



    let layers: Vec<Box<dyn Layer>> = vec![Box::new(linear1), Box::new(linear2)];
    let activations: Vec<Box<dyn ActivationFunction>>  = vec![Box::new(activation1), Box::new(activation2)];
    let loss = Box::new(CrossEntropy::new());

    let model = SequentialNeuralNetwork::new(layers, activations, loss);
    let lr = 0.01;
    let beta = 0.0;
    let mut optim = SGD::new(model, lr, beta); // Give ownership of model to optim

    // Create some training data
    let x = DMatrix::from_row_slice(4, 2, &[-4.0, -3.0,
                                            11.8, 3.2,
                                            -7.13, 1.56,
                                            0.132, 4.5896]);
    let y = DMatrix::from_row_slice(4, 2, &[0.0, 1.0,
                                            0.0, 1.0,
                                            1.0, 0.0,
                                            0.0, 1.0]);

    // Training loop
    for _ in 0..10000 {
        optim.update(&x, &y);
    }

    //Test the neural network model ... on the training data
    let y_pred = optim.model.forward(&x).map(|x| x.round());

    println!("y_pred: {}", y_pred);
}

