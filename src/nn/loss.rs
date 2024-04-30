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
use nalgebra::{DMatrix};

// Generic trait for loss functions. Defines the forward and backward methods.
pub trait LossFunction {
    // Computes the loss given the model prediction A and the desired output Y.
    fn forward(&mut self, A: &DMatrix<f64>, Y: &DMatrix<f64>) -> f64;

    // Computes the gradient of the loss with respect to the model prediction A.
    fn backward(&mut self) -> DMatrix<f64>;
}


// Mean Squared Error Loss
pub struct MSE {
    A: DMatrix<f64>, // Model prediction, size N x C
    Y: DMatrix<f64>, // Desired output, size N x C
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
}

impl LossFunction for MSE {
    // MSE Loss = 1/N * Σ_i (A_i - Y_i)^2
    fn forward(&mut self, A: &DMatrix<f64>, Y: &DMatrix<f64>) -> f64 {
        self.A = A.clone();
        self.Y = Y.clone();

        // Initialize N and C if not already initialized
        if self.N == 0 || self.C == 0 {
            self.N = A.nrows();
            self.C = A.ncols();
            self.l_N = DMatrix::from_element(self.N, 1, 1.0);
            self.l_C = DMatrix::from_element(self.C, 1, 1.0);
        }

        // Calculate the sum of squared errors
        let square_error = (&self.A - &self.Y).component_mul(&(&self.A - &self.Y));

        // The first pre multiplication with l_N sums across rows.
        // Then, the post multiplication of this product with l_N sums
        //  the row sums across columns to give the final sum as a single number.
        let sum_square_error = &self.l_N.transpose() * &square_error * &self.l_C;
        assert!(sum_square_error.nrows() == 1 && sum_square_error.ncols() == 1);
        let loss = sum_square_error[(0, 0)] / (self.N * self.C) as f64;
        return loss;
    }

    // dLdA = 2 * (A - Y) / (N * C)
    fn backward(&mut self) -> DMatrix<f64> {
        let dLdA = 2.0 * (&self.A - &self.Y) / (self.N * self.C) as f64;
        return dLdA;
    }
}

pub struct CrossEntropy {
    A: DMatrix<f64>, // Model prediction, size N x C
    Y: DMatrix<f64>, // Desired output, size N x C
    N: usize, // Batch size
    C: usize, // Number of features in each sample
    l_N: DMatrix<f64>, // Column vector of ones of size N (N x 1)
    l_C: DMatrix<f64> // Column vector of ones of size C (C x 1)
}

impl CrossEntropy {
    pub fn new() -> Self {
        CrossEntropy {
            A : DMatrix::zeros(0, 0),
            Y : DMatrix::zeros(0, 0),
            N : 0,
            C : 0,
            l_N : DMatrix::zeros(0, 0),
            l_C : DMatrix::zeros(0, 0)
        }
    }

    // Use softmax function to transform the raw model outputs A into a
    // probability distribution consisting of C classes proportional to
    // the exponentials of the input numbers.
    pub fn softmax(&self, A: &DMatrix<f64>) -> DMatrix<f64> {
        let mut A_softmax = DMatrix::zeros(A.nrows(), A.ncols());
        for i in 0..A.nrows() {
            let row = A.row(i);
            let max_val = row.max();
            let exp_sum: f64 = row.iter().map(|x| (x - max_val).exp()).sum();
            for j in 0..A.ncols() {
                A_softmax[(i, j)] = (A[(i, j)] - max_val).exp() / exp_sum;
            }
        }
        return A_softmax;
    }
}

impl LossFunction for CrossEntropy {
    // CorssEntropyLoss(A, Y) = -Y * log(softmax(A)) * l_C
    fn forward(&mut self, A: &DMatrix<f64> , Y: &DMatrix<f64>) -> f64 {
        self.A = A.clone();
        self.Y = Y.clone();

        // Initialize N and C if not already initialized
        if self.N == 0 || self.C == 0 {
            self.N = A.nrows();
            self.C = A.ncols();
            self.l_N = DMatrix::from_element(self.N, 1, 1.0);
            self.l_C = DMatrix::from_element(self.C, 1, 1.0);
        }

        // Apply the softmax function to the model predictions
        let A_softmax = self.softmax(&self.A);

        // Calculate the negative log likelihood
        let loss = -(self.Y.component_mul(&A_softmax.map(|x| x.ln()))) * &self.l_C;

        let mean_loss = (self.l_N.transpose() * loss.clone()) / (self.N as f64);
        assert!(mean_loss.nrows() == 1 && mean_loss.ncols() == 1);

        return mean_loss[(0, 0)];
    }

    // dLdA = (softmax(A) - Y) / N
    fn backward(&mut self) -> DMatrix<f64> {
        let A_softmax = self.softmax(&self.A);
        let dLdA = (A_softmax - &self.Y) / (self.N as f64);
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

    #[test]
    fn test_cross_entropy_forward() {
        let mut cross_entropy = CrossEntropy::new();
        let A = DMatrix::from_row_slice(4, 2, &[-4., -3.,
                                                -2., -1.,
                                                 0., 1.,
                                                 2., 3.]);
        let Y = DMatrix::from_row_slice(4, 2, &[0., 1.,
                                                1., 0.,
                                                1., 0.,
                                                0., 1.]);
        let loss = cross_entropy.forward(&A, &Y);
        let expected_loss = 0.8133;
        assert_abs_diff_eq!(loss, expected_loss, epsilon = 1e-4);
    }

    #[test]
    fn test_cross_entropy_backward() {
        let mut cross_entropy = CrossEntropy::new();
        let A = DMatrix::from_row_slice(4, 2, &[-4., -3.,
                                                -2., -1.,
                                                 0., 1.,
                                                 2., 3.]);
        let Y = DMatrix::from_row_slice(4, 2, &[0., 1.,
                                                1., 0.,
                                                1., 0.,
                                                0., 1.]);
        let _ = cross_entropy.forward(&A, &Y);
        let dLdA = cross_entropy.backward();
        println!("dLdA: {}", dLdA.map(|x| (x * 1000.0).round() / 1000.0));
        let expected_dLdA = DMatrix::from_row_slice(4, 2, &[0.067, -0.067,
                                                            -0.183, 0.183,
                                                            -0.183, 0.183,
                                                            0.067, -0.067]);
        assert_abs_diff_eq!(dLdA, expected_dLdA, epsilon = 1e-3);
    }

}