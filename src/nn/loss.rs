use nalgebra::{DMatrix};

/**
    * Loss Functions
    *
    * Loss functions are used to quantify the difference between the model's prediction
    * and the actual output. The loss function is a measure of how well the model is
    * performing. The goal of training a neural network is to minimize the loss function.
    *
    * Currently, the following loss functions are implemented:
    * 1. Mean Squared Error (MSE) - L = 1/N * Σ_i (A_i - Y_i)^2
    *
**/


// Mean Squared Error Loss
pub struct MSE {
    A: DMatrix<f64>, // Model prediction
    Y: DMatrix<f64>, // Desired output
    N: usize, // Batch size
    C: usize, // Number of features in each sample
    l_N: DMatrix<f64>, // Column vector of ones of size N (N x 1)
    l_C: DMatrix<f64> // Column vector of ones of size C (C x 1)
}

impl MSE {
    pub fn new() -> Self {
        MSE {
            A : DMatrix::zeros(0, 0),
            Y : DMatrix::zeros(0, 0),
            N : 0,
            C : 0,
            l_N : DMatrix::zeros(0, 0),
            l_C : DMatrix::zeros(0, 0)
        }
    }

    // MSE Loss = 1/N * Σ_i (A_i - Y_i)^2
    pub fn forward(&mut self, A: &DMatrix<f64>, Y: &DMatrix<f64>) -> f64 {
        self.N = A.nrows();
        self.C = A.ncols();
        self.A = A.clone();
        self.Y = Y.clone();
        self.l_N = DMatrix::from_element(self.N, 1, 1.0);
        self.l_C = DMatrix::from_element(self.C, 1, 1.0);

        // Calculate the sum of squared errors
        let square_error = (&self.A - &self.Y).component_mul(&(&self.A - &self.Y));

        // The first pre multiplication with l_N sums across rows.
        // Then, the post multiplication of this product with l_N sums
        //  the row sums across columns to give the final sum as a single number.
        let sum_square_error = &self.l_N.transpose() * &square_error * &self.l_C;
        let loss = sum_square_error[(0, 0)] / (self.N * self.C) as f64;
        return loss;
    }

    // dLdA = 2 * (A - Y) / (N * C)
    pub fn backward(&mut self) -> DMatrix<f64> {
        let dLdA = 2.0 * (&self.A - &self.Y) / (self.N * self.C) as f64;
        return dLdA;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mse_forward() {
        let mut mse = MSE::new();
        let A = DMatrix::from_row_slice(4, 2, &[-4.0, -3.0,
                                                -2.0, -1.0,
                                                0.0, 1.0,
                                                2.0, 3.0]);

        let Y = DMatrix::from_row_slice(4, 2, &[0.0, 1.0,
                                                1.0, 0.0,
                                                1.0, 0.0,
                                                0.0, 1.0]);
        let loss = mse.forward(&A, &Y);
        assert_abs_diff_eq!(loss, 6.5, epsilon = 1e-8);

    }

    #[test]
    fn test_mse_backward() {
        let mut mse = MSE::new();
        let A = DMatrix::from_row_slice(4, 2, &[-4.0, -3.0,
                                                -2.0, -1.0,
                                                0.0, 1.0,
                                                2.0, 3.0]);

        let Y = DMatrix::from_row_slice(4, 2, &[0.0, 1.0,
                                                1.0, 0.0,
                                                1.0, 0.0,
                                                0.0, 1.0]);
        let _ = mse.forward(&A, &Y);
        let dLdA = mse.backward();
        let expected_dLdA = DMatrix::from_row_slice(4, 2, &[-1.0, -1.0,
                                                            -0.75, -0.25,
                                                            -0.25, 0.25,
                                                            0.5, 0.5]);

        assert_abs_diff_eq!(dLdA, expected_dLdA, epsilon = 1e-8);
    }
}