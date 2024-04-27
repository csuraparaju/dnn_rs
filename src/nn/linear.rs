use nalgebra::{DMatrix};

// Defines a simple linear layer with no activation function.

pub struct Linear {
    W : DMatrix<f64>, // Weights (C_out x C_in)
    b : DMatrix<f64>, // Bias (C_out x 1)
    A : DMatrix<f64>, // Layer input (pre-activation) (N x C_in)
    dLdW : DMatrix<f64>, // Gradient of the loss with respect to W
    dLdb : DMatrix<f64>, // Gradient of the loss with respect to b
    N : usize, // Batch size (number of samples)
    l_N : DMatrix<f64>, // Column vector of ones of size N (N x 1). Used to broadcast bias vector b.
}

impl Linear {
    // Constructor for the Linear struct. Creates a new Linear layer with
    // C_in input features and C_out output features.
    pub fn new(input_size : usize, output_size : usize) -> Self {
        Linear {
            W : DMatrix::zeros(output_size, input_size),
            b : DMatrix::zeros(output_size, 1),
            A : DMatrix::zeros(0, 0),
            dLdW : DMatrix::zeros(0, 0),
            dLdb : DMatrix::zeros(0, 0),
            N : 0,
            l_N : DMatrix::zeros(0, 0)
        }
    }

    // During forward propagation, we apply a linear transformation
    // to the incoming data A to obtain output data Z using a weight matrix
    // W and a bias vector b. That is, Z = A * W^T + ι_N * b. The variable
    // ι_N is a column vector of ones of size N (the batch size), and is used
    // to broadcast the bias vector b across all samples in the batch.
    pub fn forward(&mut self, A : &DMatrix<f64>) -> DMatrix<f64> {
        self.N = A.nrows();
        self.A = A.clone();
        self.l_N = DMatrix::from_element(self.N, 1, 1.0);
        let Z = &self.A * self.W.transpose() + &self.l_N * self.b.transpose();
        return Z; // Z has shape N x C_out
    }

    // During backward propagation, we compute the gradients of the loss with
    // respect to pre-activation input (A), the weights W and bias b. Given ∂L/∂Z
    // we can compute ∂L/∂A, ∂L/∂W and ∂L/∂b as follows:
    // ∂L/∂A = ∂L/∂Z * W
    // ∂L/∂W = (∂L/∂Z)^T * A
    // ∂L/∂b = (∂L/∂Z)^T * ι_N
    pub fn backward(&mut self, dLdZ : &DMatrix<f64>) -> DMatrix<f64> {
        let dLdA = dLdZ * &self.W;
        self.dLdW = dLdZ.transpose() * &self.A;
        self.dLdb = dLdZ.transpose() * &self.l_N;
        return dLdA;
    }

    // Helpful debug method to print the weights and biases of the layer.
    pub fn print_layer_params(&self) {
        println!("Linear Layer Parameters:");
        println!("Input Features (C_in): {}", self.W.ncols());
        println!("Output Features (C_out): {}", self.W.nrows());
        println!("Weights (W):");
        println!("{:?}", self.W);
        println!("Biases (b):");
        println!("{:?}", self.b);
    }
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_linear_forward() {
        let mut linear = Linear::new(2, 3);
        linear.W = DMatrix::from_row_slice(3, 2, &
                                                 [-2.0, -1.0, 0.0,
                                                 1.0, 2.0, 3.0]);
        linear.b = DMatrix::from_row_slice(3, 1, &[-1.0, 0.0, 1.0]);

        let A = DMatrix::from_row_slice(4, 2, &[-4.0, -3.0,
                                                -2.0, -1.0,
                                                0.0, 1.0,
                                                2.0, 3.0]);
        let Z = linear.forward(&A);
        let expected_Z = DMatrix::from_row_slice(4, 3, &[10.0, -3.0, -16.0,
                                                         4.0, -1.0, -6.0,
                                                         -2.0, 1.0, 4.0,
                                                         -8.0, 3.0, 14.0]);
        assert_abs_diff_eq!(Z, expected_Z, epsilon = 1e-12);
    }

    #[test]
    fn test_linear_backward() {
        let mut linear = Linear::new(2, 3);
        linear.W = DMatrix::from_row_slice(3, 2, &
                                                 [-2.0, -1.0, 0.0,
                                                 1.0, 2.0, 3.0]);
        linear.b = DMatrix::from_row_slice(3, 1, &[-1.0, 0.0, 1.0]);

        let A = DMatrix::from_row_slice(4, 2, &[-4.0, -3.0,
                                                -2.0, -1.0,
                                                0.0, 1.0,
                                                2.0, 3.0]);
        let _ = linear.forward(&A);

        let dLdZ = DMatrix::from_row_slice(4, 3, &[-4.0, -3.0, -2.0, -1.0,
                                                    0.0, 1.0, 2.0, 3.0,
                                                    4.0, 5.0, 6.0, 7.0]);

        let dLdA = linear.backward(&dLdZ);
        let expected_dLdA = DMatrix::from_row_slice(4, 2, &[4.0, -5.0,
                                                            4.0, 4.0,
                                                            4.0, 13.0,
                                                            4.0, 22.0]);
        assert_abs_diff_eq!(dLdA, expected_dLdA, epsilon = 1e-12);

        let expected_dLdW = DMatrix::from_row_slice(3, 2, &[28.0, 30.0,
                                                            24.0, 30.0,
                                                            20.0, 30.0]);
        let expected_dLdb = DMatrix::from_row_slice(3, 1, &[2.0,
                                                            6.0,
                                                            10.0]);
        assert_abs_diff_eq!(linear.dLdW, expected_dLdW, epsilon = 1e-12);
        assert_abs_diff_eq!(linear.dLdb, expected_dLdb, epsilon = 1e-12);
    }

}