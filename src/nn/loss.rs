use nalgebra::{DMatrix};

pub enum Criterion {
    // Mean Squared Error Loss
    MSE(MSE)
    // TODO: Implement more loss functions
}

impl Criterion {
    // The Forward method takes in model prediction A and desired output
    // Y of the same shape to calculate and return a loss value L.
    // The loss value is a scalar quantity used to quantify the mismatch
    // between the network output and the desired output.
    pub fn forward(&mut self, y_pred : &DMatrix<f64>, y_true : &DMatrix<f64>) -> f64 {
        match self {
            Criterion::MSE(mse) => mse.forward(y_pred, y_true),
        }
    }

    // The Backward method calculates and returns dLdA, how changes in
    // model outputs A affect loss L. It is used to enable downstream computation,
    // as seen in previous sections.
    pub fn backward(&mut self) -> DMatrix<f64> {
        match self {
            Criterion::MSE(mse) => mse.backward(),
        }
    }
}

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
        println!("self.A: {:?}", self.A);
        println!("self.Y: {:?}", self.Y);
        println!("self.N: {:?}", self.N);
        println!("self.C: {:?}", self.C);


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
        let mut mse = Criterion::MSE(MSE::new());
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
        let mut mse = Criterion::MSE(MSE::new());
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