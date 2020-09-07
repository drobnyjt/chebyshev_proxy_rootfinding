#[macro_use(s)]
extern crate ndarray;

#[cfg(test)]
mod tests {

    use crate::chebyshev::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    fn f(x: f64) -> f64 {
        (x - 2.)*(x + 3.)
    }

    #[test]
    fn test_chebyshev() {
        let a = -10.;
        let b = -10.;

        let (intervals, coefficients) = chebyshev_subdivide(&f, vec![(a, b)], 5, 1E-3, 100);
    }
}

mod chebyshev {
    use ndarray::{Array2, Array1, ArrayBase};
    //use ndarray_linalg::*;
    use std::f64::consts::PI;

    fn p(j: usize, N: usize) -> f64 {
        if (j == 0) || (j == N) {
            2.
        } else {
            1.
        }
    }

    fn delta(j: usize, k: usize) -> f64 {
        if j == k {
            1.
        } else {
            0.
        }
    }

    fn interpolation_matrix(N: usize) -> Array2<f64> {

        let mut I_jk: Array2<f64> = Array2::zeros((N + 1, N + 1));

        for j in 0..N + 1 {
            for k in 0..N + 1 {
                I_jk[[j, k]] = 2./p(j, N)/p(k, N)/N as f64*(j as f64*PI*k as f64/N as f64).cos();
            }
        }

        return I_jk
    }

    fn chebyshev_coefficients(f: &dyn Fn(f64) -> f64, a: f64, b: f64, N: usize) -> Array1<f64> {

        let xk = lobatto_grid(a, b, N);
        let I_jk = interpolation_matrix(N);
        let f_xk: Array1<f64> = xk.iter().map(|x| f(*x)).collect();
        let a_j = I_jk.dot(&f_xk);

        return a_j
    }

    fn lobatto_grid(a: f64, b: f64, N: usize) -> Vec<f64> {

        let mut xk: Vec<f64> = vec![0.; N + 1];

        for k in 0..N + 1 {
            xk[k] = (b - a)/2.*(PI*k as f64/N as f64).cos() + (b + a)/2.;
        }

        return xk
    }

    pub fn chebyshev_subdivide(f: &dyn Fn(f64) -> f64, intervals: Vec<(f64, f64)>, N0: usize, epsilon: f64, N_max: usize) -> (Vec<(f64, f64)>, Vec<Array1<f64>>) {
        let mut coefficients: Vec<Array1<f64>> = Vec::new();
        let mut intervals_out: Vec<(f64, f64)> = Vec::new();

        for interval in intervals {
            let a = interval.0;
            let b = interval.1;

            let (a_0, error) = chebyshev_adapative(f, a, b, N0, epsilon, N_max);

            if error < epsilon {
                intervals_out.push(interval);
                coefficients.push(a_0);
            } else {
                let a1 = a;
                let b1 = a + (b - a)/2.;

                let a2 = a + (b - a)/2.;
                let b2 = b;

                let (intervals_new, coefficients_new) = chebyshev_subdivide(f, vec![(a1, b1), (a2, b2)], N0, epsilon, N_max);

                for (i, c) in intervals_new.iter().zip(coefficients_new) {
                    intervals_out.push(i.clone());
                    coefficients.push(c.clone());
                }
            }
        }
        return (intervals_out, coefficients)
    }

    fn chebyshev_approximate(a_j: Array1<f64>, a: f64, b: f64, x: f64) -> f64 {

        let N = a_j.len() - 1;

        let xi = (2.*x - (b + a))/(b - a);
        let mut b0 = 0.;
        let mut b1 = 0.;
        let mut b2 = 0.;
        let mut b3 = 0.;

        for i in 1..N + 1 {
            b0 = 2.*xi*b1 - b2 + a_j[N - i];
            b3 = b2;
            b2 = b1;
            b1 = b0;
        }

        (b0 - b3 + a_j[0])/2.
    }

    fn chebyshev_adapative(f: &dyn Fn(f64) -> f64, a: f64, b: f64, N0: usize, epsilon: f64, N_max: usize) -> (Array1<f64>, f64) {

        let mut a_0 = chebyshev_coefficients(f, a, b, N0);
        let mut N0 = N0;

        while true {

            let N1 = 2*N0;
            let a_1 = chebyshev_coefficients(f, a, b, N1);

            let error = a_0.iter().enumerate().map(|(i, a)| (a - a_1[i]).abs()).sum::<f64>() + a_1.slice(s![N0 + 1..]).iter().map(|a| a.abs()).sum::<f64>();

            if a_1[a_1.len() - 1] == 0. {
                return (a_0, 2.*N0 as f64)
            }

            if (error < epsilon) || (2*N1 - 1 >= N_max) {
                return (a_1, error)
            }

            a_0 = a_1;
            N0 = N1;
        }

        (a_0, 2.*N0 as f64)
    }
}
