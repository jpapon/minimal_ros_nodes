#ifndef TRACKING_IMPL_ADAPTIVE_PARTICLE_OMP_FILTER_H_
#define TRACKING_IMPL_ADAPTIVE_PARTICLE_OMP_FILTER_H_

#include <tracker/adaptive_particle_filter_omp.h>
template <typename PointInT, typename StateT> bool
pcl::tracking::AdaptiveParticleFilterOMPTracker<PointInT, StateT>::initCompute ()
{
  if (!Tracker<PointInT, StateT>::initCompute ())
  {
    PCL_ERROR ("[pcl::%s::initCompute] Init failed.\n", getClassName ().c_str ());
    return (false);
  }
  
  if (!particles_ || particles_->points.empty ())
    initParticles (true);
  
  return true;
}


template <typename PointInT, typename StateT> void
pcl::tracking::AdaptiveParticleFilterOMPTracker<PointInT, StateT>::weight ()
{
 // ++frame_count;
  //pcl::StopWatch timer;
 // double t_start = timer.getTime ();
 // #ifdef _OPENMP
 // #pragma omp parallel for num_threads(threads_) schedule(static, 10)
 // #endif
/*  for (int i = 0; i < particle_num_; i++){
      
    const Eigen::Affine3f trans = toEigenMatrix (particles_->points[i]);
    // destructively assigns to cloud
    pcl::transformPointCloud<PointInT> (ref_,  *transed_reference_vector_[i], trans);
  }*/
  //PointCloudInPtr coherence_input (new PointCloudIn);
  //this->cropInputPointCloud (input_, *coherence_input);
  //changed_ = true;
  //coherence_->setTargetCloud (coherence_input);
  //coherence_->initCompute ();
  //std::cout << "        Coherence        \n";
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(threads_) schedule(static, 10)
  #endif
  for (int i = 0; i < particle_num_; i++)
  {
    const Eigen::Affine3f trans = this->toEigenMatrix (particles_->points[i]);
    rejective_coherence_->computeSampledCoherence (ref_, trans, particles_->points[i].weight);
  }
  
  int num_zero = 0;
  for (int i = 0; i < particle_num_; i++)
  {
    if ( particles_->points[i].weight == 0)
      ++num_zero;
  }
  //std::cout << "=========     NUM ZERO = "<<num_zero<<"   size="<<particle_num_<<"    ====\n";
  normalizeWeight ();
  //T1 += timer.getTime () - t_start;
  //if (frame_count % 10 ==0)
  //  std::cout<< "==   size="<<this->ref_->size ()<<"   threads="<<threads_<<"   ===  "<<T1/frame_count<<"   ====="<<std::endl;
}

template <typename PointInT, typename StateT> void
pcl::tracking::AdaptiveParticleFilterOMPTracker<PointInT, StateT>::update ()
{
  //std::cout << "UPDATE\n";
  
  StateT orig_representative = representative_state_;
  representative_state_.zero ();
  representative_state_.weight = 0.0;
  float min_weight = 0.1f* 1.0f / static_cast<float> (particles_->points.size ());
  /*
  std::vector <int> valid_indices;
  valid_indices.reserve (particles_->points.size ());
  float weight_sum = 0.0f;
  for ( size_t i = 0; i < particles_->points.size (); i++)
  {
    StateT p = particles_->points[i];
    if (p.weight > min_weight)
    {
      valid_indices.push_back (i);
      weight_sum += p.weight;
    }
  }*/
  //std::cout << "Using "<<valid_indices.size ()<<" particles out of "<<this->getParticleNum ()<< "              \n";
  for ( size_t i = 0; i < particles_->points.size (); i++)
  {
    StateT p = particles_->points[i];
    representative_state_ = representative_state_ + p * p.weight;
  }
  representative_state_.weight = 1.0f / static_cast<float> (particles_->points.size ());
  motion_ = (representative_state_ - orig_representative) * (1.0 / delta_t_);
    
}

template <typename PointInT, typename StateT> void
pcl::tracking::AdaptiveParticleFilterOMPTracker<PointInT, StateT>::setInputCloud (const CloudInTConstPtr &cloud, double t_stamp)
{
  if (last_stamp_ == 0)
    last_stamp_ = t_stamp - 0.1;
    
  delta_t_ = t_stamp - last_stamp_;
  Tracker<PointInT, StateT>::setInputCloud (cloud);
  last_stamp_ = t_stamp;
}

template <typename PointInT, typename StateT> void
pcl::tracking::AdaptiveParticleFilterOMPTracker<PointInT, StateT>::resample ()
{
  //std::cout << "RESAMPLE\n";
  unsigned int k = 0;
  unsigned int n = 0;
  PointCloudStatePtr S (new PointCloudState);
  S->points.reserve (particle_num_);

  std::vector<std::vector<int> > B; // bins
  
  // initializing for sampling without replacement
  std::vector<int> a (particles_->points.size ());
  std::vector<double> q (particles_->points.size ());
  this->genAliasTable (a, q, particles_);
  
  const std::vector<double> zero_mean (StateT::stateDimension (), 0.0);
  
  // select the particles with KLD sampling
  StateT delta_x = motion_ * delta_t_;
  double motion_mean = 0.5;
  double motion_sigma = 0.1;
  do
  {
    int j_n = sampleWithReplacement (a, q);
    StateT x_t = particles_->points[j_n];

    
    x_t.sample (zero_mean, step_noise_covariance_);
    
    // motion 
    if (rand () / double (RAND_MAX) < motion_ratio_)
      x_t = x_t + delta_x;
    //x_t = x_t + motion_;
    S->points.push_back (x_t);

    // calc bin
    std::vector<int> bin (StateT::stateDimension ());
    for (int i = 0; i < StateT::stateDimension (); i++)
      bin[i] = static_cast<int> (x_t[i] / bin_size_[i]);
    
    // calc bin index... how?
      if (insertIntoBins (bin, B))
        ++k;
      ++n;
  }
  while (n < maximum_particle_number_ && (k < 2 || n < calcKLBound (k)));
  
  particles_.swap (S);               // swap
  particle_num_ = static_cast<int> (particles_->points.size ());
   
}
#define PCL_INSTANTIATE_AdaptiveParticleFilterOMPTracker(T,ST) template class PCL_EXPORTS pcl::tracking::AdaptiveParticleFilterOMPTracker<T,ST>;

#endif
