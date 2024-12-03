use rand::prelude::*;
use rand_distr::{Exp1, StandardNormal};
use crate::geometry::{ThreeVector, FourVector};
use super::{Species, Particle};
use super::dstr::RadialDistribution;

#[derive(Copy,Clone)]
pub struct BeamBuilder {
    species: Species,
    num: usize,
    weight: f64,
    normal_espec: Option<bool>,
    custom_espec: Option<bool>,
    raw_particles: Option<bool>,
    gamma: f64,
    sigma: f64,
    gamma_min: f64,
    gamma_max: f64,
    radial_dstr: RadialDistribution,
    sigma_z: f64,
    energy_chirp: f64,
    angle: f64,
    rms_div: f64,
    initial_z: f64,
    offset: ThreeVector,
}

impl BeamBuilder {
    pub fn new(species: Species, num: usize, initial_z: f64) -> Self {
        BeamBuilder {
            species,
            num,
            weight: 1.0,
            normal_espec: None,
            custom_espec: None,
            raw_particles: None,
            gamma: 0.0,
            sigma: 0.0,
            gamma_min: 0.0,
            gamma_max: 0.0,
            radial_dstr: RadialDistribution::Uniform {r_max: 0.0},
            sigma_z: 0.0,
            energy_chirp: 0.0,
            angle: 0.0,
            rms_div: 0.0,
            initial_z,
            offset: ThreeVector::new(0.0, 0.0, 0.0),
        }
    }

    pub fn with_weight(&self, weight: f64) -> Self {
        BeamBuilder {
            weight,
            ..*self
        }
    }

    pub fn with_normal_energy_spectrum(&self, gamma: f64, sigma: f64) -> Self {
        BeamBuilder {
            normal_espec: Some(true),
            custom_espec: Some(false),
            raw_particles: Some(false),
            gamma,
            sigma,
            ..*self
        }
    }

    pub fn with_bremsstrahlung_spectrum(&self, gamma_min: f64, gamma_max: f64) -> Self {
        BeamBuilder {
            normal_espec: Some(false),
            custom_espec: Some(false),
            raw_particles: Some(false),
            gamma_min,
            gamma_max,
            ..*self
        }
    }

    pub fn with_custom_spectrum(&self) -> Self {
        BeamBuilder {
            normal_espec: Some(false),
            custom_espec: Some(true),
            raw_particles: Some(false),
            ..*self
        }
    }

    pub fn with_raw_particles(&self) -> Self {
        BeamBuilder {
            normal_espec: Some(false),
            custom_espec: Some(false),
            raw_particles: Some(true),
            ..*self
        }
    }

    pub fn with_divergence(&self, rms_div: f64) -> Self {
        BeamBuilder {
            rms_div,
            ..*self
        }
    }

    pub fn with_collision_angle(&self, angle: f64) -> Self {
        BeamBuilder {
            angle,
            ..*self
        }
    }

    pub fn with_normally_distributed_xy(&self, sigma_x: f64, sigma_y: f64) -> Self {
        BeamBuilder {
            radial_dstr: RadialDistribution::Normal { sigma_x, sigma_y },
            ..*self
        }
    }

    pub fn with_trunc_normally_distributed_xy(&self, sigma_x: f64, sigma_y: f64, x_max: f64, y_max: f64) -> Self {
        BeamBuilder {
            radial_dstr: RadialDistribution::TruncNormal { sigma_x, sigma_y, x_max, y_max },
            ..*self
        }
    }

    pub fn with_uniformly_distributed_xy(&self, r_max: f64) -> Self {
        BeamBuilder {
            radial_dstr: RadialDistribution::Uniform { r_max },
            ..*self
        }
    }

    pub fn with_length(&self, sigma_z: f64) -> Self {
        BeamBuilder {
            sigma_z,
            ..*self
        }
    }

    pub fn with_offset(&self, offset: ThreeVector) -> Self {
        BeamBuilder {
            offset,
            ..*self
        }
    }

    pub fn with_energy_chirp(&self, rho: f64) -> Self {
        BeamBuilder {
            energy_chirp: rho,
            ..*self
        }
    }

    pub fn build<R: Rng>(&self, rng: &mut R, spectrum_file: String, id: i32,  raw_x: Vec<f64>, raw_y: Vec<f64>, raw_z: Vec<f64>, raw_px: Vec<f64>, raw_py: Vec<f64>, raw_pz: Vec<f64>, raw_weight: Vec<f64>, start_ind: Vec<usize>) -> Vec<Particle> {
        let normal_espec = self.normal_espec.expect("primary energy spectrum not specified");
        let custom_espec = self.custom_espec.expect("custom energy spectrum not specified");
        let raw_particles = self.raw_particles.expect("raw particles not specified");
        let mut gamma_ax: Vec<f64>;
        let mut n_ax: Vec<f64>;
        let max_n_ax: f64;
        let id = id as usize;

        if raw_particles {
            (0..self.num).into_iter()
                .map(|i| {

                    // need to multipy px and pz by -1 because the data is loaded in beam coordinates, however calculations done in laser coordinates
                    let u = FourVector::new(0.0, -1.0*raw_px[start_ind[id]+i], raw_py[start_ind[id]+i], -1.0*raw_pz[start_ind[id]+i]).unitize();

                    let (t, z) = if self.offset[2] >= 0.0 {
                        // beam is further away
                        (-self.initial_z, self.initial_z + self.offset[2] + raw_z[start_ind[id]+i])
                    } else {
                        // beam is closer to focal plane, push backwards
                        (-self.initial_z - self.offset[2].abs(), self.initial_z + raw_z[start_ind[id]+i])
                    };

                    let (x, y) = (-1.0*raw_x[start_ind[id]+i] + self.offset[0], raw_y[start_ind[id]+i] + self.offset[1]);
                    let r = ThreeVector::new(x, y, z);
                    let r = r.rotate_around_y(self.angle);
                    let r = FourVector::new(t, r[0], r[1], r[2]);

                    Particle::create(self.species, r)
                        .with_normalized_momentum(u)
                        .with_optical_depth(rng.sample(Exp1))
                        .with_weight(raw_weight[start_ind[id]+i])
                        .with_id(i as u64)
                        .with_parent_id(i as u64)

                })
            .collect()

        } else {

            if custom_espec {
                use std::fs::File;
                use std::path::Path;

                let file_path = Path::new(&spectrum_file);
                let display = file_path.display();

                let file = match File::open(&file_path) {
                    Err(why) => panic!("couldn't open {}: {}", display, why),
                    Ok(file) => file,
                };

                let mut rdr = csv::Reader::from_reader(file);
                gamma_ax = Vec::new();
                n_ax = Vec::new();

                for result in rdr.records() {

                    let record = result.expect("record");
                    gamma_ax.push(record[0].parse().unwrap());
                    n_ax.push(record[1].parse().unwrap());
                }
                max_n_ax = n_ax.iter().cloned().fold(0./0., f64::max);
            } else {
                gamma_ax = Vec::new();
                n_ax = Vec::new();
                max_n_ax = 1.0;
            }

            // loop starts here
            (0..self.num).into_iter()
                .map(|i| {
                    // Sample gamma from relevant distribution
                    let (gamma, dz) = if normal_espec {
                        loop {
                            // for correlated gamma and z
                            let rho = -self.energy_chirp;
                            let n0 = rng.sample::<f64,_>(StandardNormal);
                            let n1 = rng.sample::<f64,_>(StandardNormal);
                            let n2 = rho * n0 + (1.0 - rho * rho).sqrt() * n1;

                            let dz = self.sigma_z * n0;
                            let gamma = self.gamma + self.sigma * n2;
                            if gamma > 1.0 {
                                break (gamma, dz);
                            }
                        }
                    } else if custom_espec {
                        loop {
                            let ax_length = gamma_ax.len() as f64;
                            let x = rng.gen::<f64>() * (ax_length - 1.0);
                            let x_int = x as i64;
                            let x_rem = x - (x_int as f64);
                            let x_ind = x_int as usize;
                            //let x = (rng.gen::<f64>() * (ax_length - 1.0)).round() as usize;
                            let u = rng.gen::<f64>() * max_n_ax;

                            if u <= (n_ax[x_ind] + (n_ax[x_ind + 1] - n_ax[x_ind]) * x_rem) {
                                let dz = self.sigma_z * rng.sample::<f64,_>(StandardNormal);
                                let gamma = gamma_ax[x_ind] + (gamma_ax[x_ind + 1] - gamma_ax[x_ind]) * x_rem;
                                break (gamma, dz);
                            }
                        }
                    } else { // brem spec
                        let x_min = self.gamma_min / self.gamma_max;
                        let y_max = 4.0 / (3.0 * x_min) - 4.0 / 3.0 + x_min;
                        let x = loop {
                            let x = x_min + (1.0 - x_min) * rng.gen::<f64>();
                            let u = rng.gen::<f64>();
                            let y = 4.0 / (3.0 * x) - 4.0 / 3.0 + x;
                            if u <= y / y_max {
                                break x;
                            }
                        };

                        let dz = self.sigma_z * rng.sample::<f64,_>(StandardNormal);
                        (x * self.gamma_max, dz)
                    };

                    let u = match self.species {
                        Species::Electron | Species::Positron => -(gamma * gamma - 1.0).sqrt(),
                        Species::Photon => -gamma,
                    };

                    let theta_x = self.angle + self.rms_div * rng.sample::<f64,_>(StandardNormal);
                    let theta_y = self.rms_div * rng.sample::<f64,_>(StandardNormal);

                    let u = match self.species {
                        Species::Electron | Species::Positron => FourVector::new(0.0, u * theta_x.sin() * theta_y.cos(), u * theta_y.sin(), u * theta_x.cos() * theta_y.cos()).unitize(),
                        Species::Photon => FourVector::lightlike(u * theta_x.sin() * theta_y.cos(), u * theta_y.sin(), u * theta_x.cos() * theta_y.cos()),
                    };

                    let (t, z) = if self.offset[2] >= 0.0 {
                        // beam is further away
                        (-self.initial_z, self.initial_z + self.offset[2] + dz)
                    } else {
                        // beam is closer to focal plane, push backwards
                        (-self.initial_z - self.offset[2].abs(), self.initial_z + dz)
                    };

                    let (x, y) = self.radial_dstr.sample(rng);

                    let (x, y) = (x + self.offset[0], y + self.offset[1]);
                    let r = ThreeVector::new(x, y, z);
                    let r = r.rotate_around_y(self.angle);
                    let r = FourVector::new(t, r[0], r[1], r[2]);

                    Particle::create(self.species, r)
                        .with_normalized_momentum(u)
                        .with_optical_depth(rng.sample(Exp1))
                        .with_weight(self.weight)
                        .with_id(i as u64)
                        .with_parent_id(i as u64)
                })
            .collect()
        }
    }
}
