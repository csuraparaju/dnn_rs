use nalgebra::{DMatrix};
use std::f64::consts;

/**
    * Activation Functions
    *
    * Activation functions are used to introduce non-linearity in the neural network.
    * The primary purpose of having nonlinear components in the neural network (fNN )
    * is to allow it to approximate nonlinear functions. Without activation functions,
    * fNN will always be linear, no matter how deep it is.
    *
    * Currently, the following activation functions are implemented:
    * 1. Identity - f(x) = x
    * 2. ReLU - f(x) = max(0, x)
    *
**/


// Identity Activation Function
pub struct Identity {
    A : DMatrix<f64>
}

impl Identity {
    pub fn new() -> Self {
        Identity {
            A : DMatrix::zeros(0, 0)
        }
    }

    pub fn forward(&mut self, Z : &DMatrix<f64>) -> DMatrix<f64> {
        self.A = Z.clone(); // Identity(Z) = Z
        return self.A.clone();
    }

    pub fn backward(&self, dLdA : &DMatrix<f64>) -> DMatrix<f64> {
        let dLdZ = dLdA; // Derivative of Identity is 1, so dLdZ = dLdA * 1 = dLdA
        return dLdZ.clone();
    }
}

// ReLU Activation Function
pub struct ReLU {
    A : DMatrix<f64>
}

impl ReLU {
    pub fn new() -> Self {
        ReLU {
            A : DMatrix::zeros(0, 0)
        }
    }

    pub fn forward(&mut self, Z : &DMatrix<f64>) -> DMatrix<f64> {
        self.A = Z.map(|x| x.max(0.0)); // ReLU(Z) = max(0, Z)
        return self.A.clone();
    }

    pub fn backward(&self, dLdA : &DMatrix<f64>) -> DMatrix<f64> {
        // Assert that forward pass is  called before backward pass
        // to ensure that self.A is set to the correct value.
        assert!(!self.A.is_empty(), "Forward pass not called before backward pass");

        // Derivative of ReLU is 1 if x > 0, 0 otherwise
        let dAdZ = self.A.map(|x| if x > 0.0 { 1.0 } else { 0.0 });
        dLdA.component_mul(&dAdZ) // dLdZ = dLdA * dA/dZ
    }
}

// Sigmoid Activation Function
pub struct Sigmoid {
    A : DMatrix<f64>
}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid{
            A : DMatrix::zeroes(0, 0);
        }
    }
    pub fn forward(&mut self, Z : &DMatrix<f64>){
        self.A = Z.map(|x| 1\(1 + consts::E.powi(x)));
        return self.A.clone;
    }




}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_identity_forward() {
        let mut identity = Identity::new();
        let Z = DMatrix::from_row_slice(2, 2, &[1.0, 2.0,
                                                3.0, 4.0]);
        let A = identity.forward(&Z);
        assert_abs_diff_eq!(A, Z, epsilon = 1e-12);
    }

    #[test]
    fn test_identity_backward() {
        let identity = Identity::new();
        let dLdA = DMatrix::from_row_slice(2, 2, &[1.0, 2.0,
                                                    3.0, 4.0]);
        let dLdZ = identity.backward(&dLdA);
        assert_abs_diff_eq!(dLdZ, dLdA, epsilon = 1e-12);
    }

    #[test]
    fn test_relu_forward() {
        let mut relu = ReLU::new();
        let Z = DMatrix::from_row_slice(2, 3, &[0.0378, 0.3022, -1.6123,
                                                -2.5186, -1.9395, 1.4077]);
        let A = relu.forward(&Z);
        let expected = DMatrix::from_row_slice(2, 3, &[0.0378, 0.3022,
                                                       0.0, 0.0, 0.0, 1.4077]);
        assert_abs_diff_eq!(A, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_relu_backward() {
        let mut relu = ReLU::new();
        let Z = DMatrix::from_row_slice(2, 3, &[0.0378, 0.3022, -1.6123,
                                                -2.5186, -1.9395, 1.4077]);
        let _ = relu.forward(&Z);
        let dLdA = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0,
                                                   4.0, 5.0, 6.0]); // Mock dLdA
        let dLdZ = relu.backward(&dLdA);
        let expected = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 0.0,
                                                       0.0, 0.0, 6.0]);
        assert_abs_diff_eq!(dLdZ, expected, epsilon = 1e-12);
    }

}
