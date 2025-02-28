//! Classical synchrotron emission, e -> e + gamma, in a background field

use std::f64::consts;
use crate::constants::*;
use crate::pwmci;
use crate::geometry::{ThreeVector, FourVector, StokesVector};
use crate::special_functions::Airy;
use super::tables;

/// Returns the classical synchrotron rate, per unit time (in seconds)
pub fn rate(chi: f64, gamma: f64) -> f64 {
    let h = 5.0 * consts::FRAC_PI_3;
    3.0f64.sqrt() * ALPHA_FINE * chi * h / (2.0 * consts::PI * gamma * COMPTON_TIME)
}

/// Samples the classical synchrotron spectrum of an electron with
/// quantum parameter `chi` and Lorentz factor `gamma`.
/// 
/// Returns a triple of the photon energy, in units of mc^2,
/// and the polar and azimuthal angles of emission, in the range
/// [0,pi] and [0,2pi] respectively.
/// 
/// As there is no hbar-dependent cutoff, the energy of the photon
/// can exceed that of the electron.
pub fn sample(chi: f64, gamma: f64, rand1: f64, rand2: f64, rand3: f64) -> (f64, Option<f64>, f64) {
    // First determine z:
    // z^(1/3) = (2 + 4 cos(delta/3)) / (5 (1-r)) where 0 <= r < 1
    // and cos(delta) = (-9 + 50r - 25r^2) / 16
    let delta = ((-9.0 + 50.0 * rand2 - 25.0 * rand2.powi(2)) / 16.0).acos();
    let z = ((2.0 + 4.0 * (delta/3.0).cos()) / (5.0 * (1.0 - rand2))).powi(3);

    // now invert cdf(u|z) = (3/pi) \int_0^x t K_{1/3}(t) dt,
    // which is tabulated, to obtain x = 2 u z / (3 chi)
    // for x < 0.01, cdf(u|z) =

    let ln_rand = rand1.ln();
    let x = if ln_rand < tables::CLASSICAL_SPECTRUM_TABLE[0][1] {
        1.020377255 * rand1.powf(0.6)
    } else {
        //println!("Inverting ln(rand = {}) = {}", rand1, ln_rand);
        let (ln_x, _) = pwmci::invert(ln_rand, &tables::CLASSICAL_SPECTRUM_TABLE)
            .unwrap_or( (tables::CLASSICAL_SPECTRUM_TABLE.last().unwrap()[0],1) );
        ln_x.exp()
    };

    let u = 3.0 * chi * x / (2.0 * z);
    let omega_mc2 = u * gamma;

    let cos_theta = (gamma - z.powf(2.0/3.0) / (2.0 * gamma)) / (gamma.powi(2) - 1.0).sqrt();
    let theta = if cos_theta >= 1.0 {
        Some(0.0)
    } else if cos_theta >= -1.0 {
        Some(cos_theta.acos())
    } else {
        None
    };

    (omega_mc2, theta, 2.0 * consts::PI * rand3)
}

/// Returns the Stokes vector of the photon with four-momentum `k` (normalized to the
/// electron mass), assuming that it was emitted by an electron with quantum parameter `chi`,
/// Lorentz factor `gamma`, velocity `v` and instantaneous acceleration `w`.
///
/// The basis is defined with respect to a vector in the `x`-`z` plane that is perpendicular
/// to the photon three-momentum.
pub fn stokes_parameters(k: FourVector, chi: f64, gamma: f64, v: ThreeVector, w: ThreeVector) -> StokesVector {
    // belt and braces
    let v = v.normalize();
    let w = w.normalize();
    let n = ThreeVector::from(k).normalize();

    // u = omega / e
    let u = k[0] / gamma;

    // angle b/t k and plane (v, w)
    let beta = {
        let q = v.cross(w).normalize();
        (n * q).asin()
    };

    let mu = (beta * beta + 1.0 / (gamma * gamma)).sqrt();
    let eta = u * (gamma * mu).powi(3) / (3.0 * chi);

    let k1_3 = eta.bessel_K_1_3().unwrap_or(0.0);
    let k2_3 = eta.bessel_K_2_3().unwrap_or(0.0);

    let g = (beta * k1_3).powi(2) + (mu * k2_3).powi(2);

    let xi = ThreeVector::new(
        ((mu * k2_3).powi(2) - (beta * k1_3).powi(2)) / g,
        0.0,
        2.0 * beta * mu * k1_3 * k2_3 / g,
    );

    // xi is defined w.r.t. the basis [w - (n.w) n, n, w x n], whereas
    // we want [e, n, e x n], where e is in the x-z plane.
    // phi rotates w - (n.w)n down to the x-z plane - we don't care which
    // direction in the x-z plane, because this is fixed by k
    let phi = ((w - (n * w) * n) * ThreeVector::new(0.0, 1.0, 0.0)).asin();

    // which rotates the Stokes parameters through 2 phi
    let xi = ThreeVector::new(
        (2.0 * phi).cos() * xi[0] + (2.0 * phi).sin() * xi[1],
        -(2.0 * phi).sin() * xi[0] + (2.0 * phi).cos() * xi[1],
        xi[2],
    );

    [1.0, xi[0], xi[1], xi[2]].into()
}

#[cfg(test)]
mod tests {
    use std::convert::TryInto;
    use rand::prelude::*;
    use rand_xoshiro::*;
    use crate::{Particle, Species};
    use super::*;

    #[test]
    fn classical_stokes_vector() {
        let (chi, gamma) = (0.1, 1000.0);
        let w: ThreeVector = [1.0, 0.0, 0.0].into();
        let long: ThreeVector = [0.0, 0.0, 1.0].into();
        let perp: ThreeVector = [1.0, 0.0, 0.0].into();
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let target_omega_mc2 = 0.1 * gamma;
        println!("Sampling at omega/(m gamma) = {:.3e}...", target_omega_mc2 / gamma);

        // integrating over all angles is expected to yield a Stokes vector
        //   [1.0, K_{2/3}(k) / int_k^infty K_{5/3}(x) dx, 0, 0],
        // where k = 2 omega / (3 gamma m chi)
        // let sv: StokesVector = (0..20_000_000)
        let output = (0..20_000_000)
            .map(|_| sample(chi, gamma, rng.gen(), rng.gen(), rng.gen()))
            .filter(|(omega_mc2, _, _)|
                *omega_mc2 < 1.01 * target_omega_mc2 && *omega_mc2 > 0.99 * target_omega_mc2
            )
            .map(|(omega_mc2, theta, cphi)| {
                let theta = theta.unwrap();
                // println!("omega/(m gamma) = {:.2e}, gamma theta = {:.3e}, phi = {:.3e}", omega_mc2 / gamma, gamma * theta, cphi);

                // photon four-momentum
                let perp = perp.rotate_around(long, cphi);
                let k: ThreeVector = omega_mc2 * (theta.cos() * long + theta.sin() * perp);
                let k = FourVector::lightlike(k[0], k[1], k[2]);

                stokes_parameters(k, chi, gamma, long, w)
            })
            // .fold([0.0; 4].into(), |a, b| a + b);
            .fold([0.0; 7], |a, b| [
                a[0] + b[0],
                a[1] + b[1],
                a[2] + b[2],
                a[3] + b[3],
                a[4] + b[1] * b[1],
                a[5] + b[2] * b[2],
                a[6] + b[3] * b[3],
            ]);

        let count = output[0];

        // Averaged Stokes vector
        let sv: [f64; 4] = output[0..4].try_into().unwrap();
        let sv = StokesVector::from(sv) / count;

        let error = StokesVector::from([0.0, output[4], output[5], output[6]]) / count;
        let error = [0.0, (error[1] - sv[1] * sv[1]).sqrt() / count.sqrt(), (error[2] - sv[2] * sv[2]).sqrt() / count.sqrt(), (error[3] - sv[3] * sv[3]).sqrt() / count.sqrt()];

        println!("Averaged {} Stokes vectors = [\n\t{:.3e}\n\t{:.3e} ± {:.3e}\n\t{:.3e} ± {:.3e}\n\t{:.3e} ± {:.3e}\n]", count, sv[0], sv[1], error[1], sv[2], error[2], sv[3], error[3]);
        assert!(sv[2].abs() < error[2] && sv[3].abs() < error[3]);

        // Check value of sv[1]
        let xi = 2.0 * target_omega_mc2 / (3.0 * gamma * chi);

        // K(2/3, xi)
        let k2_3: f64 = GL_NODES.iter()
            .zip(GL_WEIGHTS.iter())
            .map(|(t, w)| {
                w * (-xi * t.cosh() + t).exp() * (2.0 * t / 3.0).cosh()
            })
            .sum();

        // int_xi^infty K(5/3, y) dy
        let int_k5_3: f64 = GL_NODES.iter()
            .zip(GL_WEIGHTS.iter())
            .map(|(t, w)| {
                w * (-xi * t.cosh() + t).exp() * (5.0 * t / 3.0).cosh() / t.cosh()
            })
            .sum();

        let target = k2_3 / int_k5_3;
        let diff = (sv[1] - target).abs() / target;
        println!("Got sv[1] = {:.3e}, expected {:.3e}, error = {:.3}%", sv[1], target, 100.0 * diff);
        assert!(diff < error[1]);

        // Finally, project Stokes parameters onto detector in the x-y plane
        let photon = Particle::create(Species::Photon, [0.0; 4].into())
            .with_normalized_momentum([target_omega_mc2, 0.0, 0.0, target_omega_mc2].into())
            .with_polarization(sv);

        let pol_x = photon.polarization_along_x();
        let pol_y = photon.polarization_along_y();
        println!("Projected onto x = {:.3e}, y = {:.3e}, total = {:.3e}", pol_x, pol_y, pol_x + pol_y);
        assert!((pol_x + pol_y - 1.0).abs() < 1.0e-15);
    }
}

#[allow(unused)]
static GL_NODES: [f64; 32] = [
	4.448936583326702e-2,
	2.345261095196185e-1,
	5.768846293018864e-1,
	1.072448753817818e+0,
	1.722408776444645e+0,
	2.528336706425795e+0,
	3.492213273021994e+0,
	4.616456769749767e+0,
	5.903958504174244e+0,
	7.358126733186241e+0,
	8.982940924212596e+0,
	1.078301863253997e+1,
	1.276369798674273e+1,
	1.493113975552256e+1,
	1.729245433671531e+1,
	1.985586094033605e+1,
	2.263088901319677e+1,
	2.562863602245925e+1,
	2.886210181632347e+1,
	3.234662915396474e+1,
	3.610049480575197e+1,
	4.014571977153944e+1,
	4.450920799575494e+1,
	4.922439498730864e+1,
	5.433372133339691e+1,
	5.989250916213402e+1,
	6.597537728793505e+1,
	7.268762809066271e+1,
	8.018744697791352e+1,
	8.873534041789240e+1,
	9.882954286828397e+1,
	1.117513980979377e+2,
];

#[allow(unused)]
static GL_WEIGHTS: [f64; 32] = [
	1.092183419523850e-1,
	2.104431079388132e-1,
	2.352132296698480e-1,
	1.959033359728810e-1,
	1.299837862860718e-1,
	7.057862386571744e-2,
	3.176091250917507e-2,
	1.191821483483856e-2,
	3.738816294611525e-3,
	9.808033066149551e-4,
	2.148649188013642e-4,
	3.920341967987947e-5,
	5.934541612868633e-6,
	7.416404578667552e-7,
	7.604567879120781e-8,
	6.350602226625807e-9,
	4.281382971040929e-10,
	2.305899491891336e-11,
	9.799379288727094e-13,
	3.237801657729266e-14,
	8.171823443420719e-16,
	1.542133833393823e-17,
	2.119792290163619e-19,
	2.054429673788045e-21,
	1.346982586637395e-23,
	5.661294130397359e-26,
	1.418560545463037e-28,
	1.913375494454224e-31,
	1.192248760098222e-34,
	2.671511219240137e-38,
	1.338616942106256e-42,
	4.510536193898974e-48,
];