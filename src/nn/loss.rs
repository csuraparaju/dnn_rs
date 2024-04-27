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
    pub fn backward(&self) -> DMatrix<f64> {
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
}

impl MSE {
    pub fn new() -> Self {
        MSE {
            A : DMatrix::zeros(0, 0),
            Y : DMatrix::zeros(0, 0),
            N : 0,
            C : 0
        }
    }

    pub fn forward(&self, A: &DMatrix<f64>, Y: &DMatrix<f64>) -> f64 {
        self.N = A.nrows();
        self.C = A.ncols();
        self.A = A.clone();
        self.Y = Y.clone();
        let loss = (1.0 / (self.N * self.C) as f64) * (self.A - self.Y).norm_squared();
        return loss;
    }
}