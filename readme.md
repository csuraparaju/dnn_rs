## dnn_rs: A Deep Learning Library written in Rust
A deep learning library written in Rust, structured similary to pytorch. This library is still in development and is not yet ready for use. The goal is to provide a high performance deep learning library that is easy to use and understand.

## Features (As of 4/27/2024)
- Activation Functions:
    - [x] Identity
    - [x] ReLU
- Layers:
    - [x] Linear
- Loss Functions:
    - [x] Mean Squared Error
- Model:
    - [x] Sequential model, which owns a vector of layers.
- Optimizers:
    - [x] Stochastic Gradient Descent

- [x] Modular design, with a activation, layer, loss, and optimizer module.
- [x] Each implemented type has a forward and backward function, allowing for easy backpropagation.
- [x] Uses the high performance nalgebra library for matrix operations.

## Project Structure
```The project is structured as a library, with the following modules:
├── lib.rs
├── nn
│   ├── activation.rs
│   ├── layers.rs
│   ├── loss.rs
│   └── model.rs
└── optim
    └── sgd.rs
```

The `nn` module contains the core components of the library, including the activation functions, layers, loss functions, and the model struct. The `optim` module contains the optimizers, which are used to update the weights of the model. Making use of rust's ownership system, the key design principle has model own the layers, and the layers own the weights and biases.

Notice that the model struct does not have an update function, but instead the optimizer is responsible for updating the weights of the model. This is intentional, and allows for more flexibility when creating a model. We can easily swap out optimizers, or even use multiple optimizers for different parts of the model. In the future, I plan to add more optimizers, such as Adam, RMSProp, and Adagrad.

## Set Up and Usage
To use this library, you will need to have Rust installed on your machine. You can install Rust by following the instructions on the official Rust website: https://www.rust-lang.org/tools/install

Once you have Rust installed, you can add this library as a dependency in your `Cargo.toml` file:
```toml
[dependencies]
dnn_rs = { git = "" }
```

Then you can use the library in your Rust code by importing the necessary modules:
```rust
use dnn_rs::nn::{Activation, Layer, Loss, Model};
use dnn_rs::optim::SGD;
```

## Example
Refer to the `examples` directory for an example of how to use this library. The example trains a simple 2-layer neural network with ReLU activation and Mean Squared Error loss using the Stochastic Gradient Descent optimizer.

## Contributing
Please feel free to contribute to this project! Clone the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
