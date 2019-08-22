/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;



void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 50;
  std::default_random_engine gen;
    
  normal_distribution<double> N_x_init(0, std[0]);
  normal_distribution<double> N_y_init(0, std[1]);
  normal_distribution<double> N_theta_init(0,std[2]);
  
  weights = vector<double>(num_particles);
  particles = vector<Particle>(num_particles);
    
  for(int i = 0; i < num_particles; i ++)
  {
    Particle p;
    p.id = i;
    p.x = x + N_x_init(gen);
    p.y = y + N_y_init(gen);
    p.theta = theta + N_theta_init(gen);
    
    p.weight = 1.0;
      
    particles.push_back(p);
  }
    
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  std::default_random_engine gen;
  
  //define normal distributions for sensor noise
  normal_distribution<double> N_x(0, std_pos[0]);
  normal_distribution<double> N_y(0, std_pos[1]);
  normal_distribution<double> N_theta(0, std_pos[2]);
    
  for(int i = 0; i < num_particles; i ++)
  {
    //calculate new state
    if(fabs(yaw_rate) < 0.00001)
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else{
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
      
    //add noise
    particles[i].x += N_x(gen);
    particles[i].y += N_y(gen);
    particles[i].theta += N_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
    for (int i = 0; i < observations.size(); i++){
        int matched_id = 0;
        double min_dist_square = INFINITY;
        for (int j = 0; j < predicted.size(); j++){
            double dist_square = (predicted[j].x - observations[i].x)*(predicted[j].x - observations[i].x) + (predicted[j].y - observations[i].y)*(predicted[j].y - observations[i].y);
            if (dist_square < min_dist_square){
                matched_id = predicted[j].id;
                min_dist_square = dist_square;
            }
        }
        observations[i].id = matched_id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
    
    double sensor_range_square = sensor_range * sensor_range;
    for (int i = 0; i < num_particles; i++){
        
        // get all landmarks in the sensor range
        std::vector<LandmarkObs> predictions;
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
            double delta_x = map_landmarks.landmark_list[j].x_f - particles[i].x;
            double delta_y = map_landmarks.landmark_list[j].y_f - particles[i].y;
            if (delta_x*delta_x + delta_y*delta_y < sensor_range_square){
                LandmarkObs landmark_temp;
                landmark_temp.x = map_landmarks.landmark_list[j].x_f;
                landmark_temp.y = map_landmarks.landmark_list[j].y_f;
                landmark_temp.id = map_landmarks.landmark_list[j].id_i;
                predictions.push_back(landmark_temp);
            }
        }
        
        if (predictions.size() == 0){
            particles[i].weight = 0.0;
            weights[i] = 0.0;
            continue;
        }
        
        //transform the measured landmarks from vehicle's coordinate to map's coordinate
        std::vector<LandmarkObs> transformed_obs;
        for (int k = 0; k < observations.size(); k++){
            LandmarkObs obs;
            obs.id = observations[k].id;
            obs.x = particles[i].x + observations[k].x * cos(particles[i].theta) - observations[k].y * sin(particles[i].theta);
            obs.y = particles[i].y + observations[k].x * sin(particles[i].theta) + observations[k].y * cos(particles[i].theta);
            transformed_obs.push_back(obs);
        }
        
        //associate the measured landmarks to ones in map;
        dataAssociation(predictions, transformed_obs);
        
        //update weights
        double total_weights = 1.0;
        for (int s = 0; s < transformed_obs.size(); s++){
            int index = transformed_obs[s].id - 1;
            double delta_x = map_landmarks.landmark_list[index].x_f - transformed_obs[s].x;
            double delta_y = map_landmarks.landmark_list[index].y_f - transformed_obs[s].y;
            double gauss_norm = 1/(2 * M_PI * std_landmark[0] * std_landmark[1]);
            double exponent = (delta_x * delta_x)/(2.0 * std_landmark[0] * std_landmark[0]) + (delta_y * delta_y)/(2.0 * std_landmark[1] * std_landmark[1]);
            total_weights *= gauss_norm * exp(-exponent);
        }
        particles[i].weight = total_weights;
        weights[i] = total_weights;
    }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
    std::default_random_engine gen;
    std::discrete_distribution<int> dist_weight(weights.begin(), weights.end());
    std::vector<Particle> resampled_particles;
    
    for (int i = 0; i < num_particles; i++){
        resampled_particles.push_back(particles[dist_weight(gen)]);
    }
    particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
