#ifndef TRACKING_ADAPTIVE_PARTICLE_FILTER_OMP_H_
#define TRACKING_ADAPTIVE_PARTICLE_FILTER_OMP_H_


#include <pcl/tracking/tracking.h>
#include <pcl/tracking/kld_adaptive_particle_filter.h>
#include <pcl/tracking/coherence.h>
#include "rejective_point_cloud_coherence.h"

namespace pcl
{
  namespace tracking
  {
    template <typename PointInT, typename StateT>
    class AdaptiveParticleFilterOMPTracker: public KLDAdaptiveParticleFilterTracker<PointInT, StateT>
    {
    public:
      
      
      
      using Tracker<PointInT, StateT>::tracker_name_;
      using Tracker<PointInT, StateT>::search_;
      using Tracker<PointInT, StateT>::input_;
      using Tracker<PointInT, StateT>::indices_;
      using Tracker<PointInT, StateT>::getClassName;
      using ParticleFilterTracker<PointInT, StateT>::ref_;
      using ParticleFilterTracker<PointInT, StateT>::representative_state_;
      using ParticleFilterTracker<PointInT, StateT>::motion_;
      using ParticleFilterTracker<PointInT, StateT>::initial_noise_mean_;
      using ParticleFilterTracker<PointInT, StateT>::initial_noise_covariance_;
      using ParticleFilterTracker<PointInT, StateT>::trans_;
      using ParticleFilterTracker<PointInT, StateT>::initParticles;
      
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::particles_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::change_detector_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::change_counter_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::change_detector_interval_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::use_change_detector_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::pass_x_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::pass_y_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::pass_z_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::alpha_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::changed_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::use_normal_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::particle_num_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::change_detector_filter_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::transed_reference_vector_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::normalizeWeight;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::normalizeParticleWeight;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::calcBoundingBox;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::motion_ratio_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::step_noise_covariance_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::sampleWithReplacement;

      
      typedef Tracker<PointInT, StateT> BaseClass;
      
      typedef typename Tracker<PointInT, StateT>::PointCloudIn PointCloudIn;
      typedef typename PointCloudIn::Ptr PointCloudInPtr;
      typedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;

      typedef PointCloud<PointInT> CloudInT;
      typedef boost::shared_ptr< CloudInT > CloudInTPtr;
      typedef boost::shared_ptr< const CloudInT > CloudInTConstPtr;
      
      typedef typename Tracker<PointInT, StateT>::PointCloudState PointCloudState;
      typedef typename PointCloudState::Ptr PointCloudStatePtr;
      typedef typename PointCloudState::ConstPtr PointCloudStateConstPtr;

      typedef PointCoherence<PointInT> Coherence;
      typedef boost::shared_ptr< Coherence > CoherencePtr;
      typedef boost::shared_ptr< const Coherence > CoherenceConstPtr;

      typedef pcl::tracking::RejectivePointCloudCoherence<PointInT> RejCoherence;
      typedef boost::shared_ptr< RejCoherence > RejCoherencePtr;
      typedef boost::shared_ptr< const RejCoherence > RejCoherenceConstPtr;

      /** \brief Initialize the scheduler and set the number of threads to use.
        * \param nr_threads the number of hardware threads to use (0 sets the value back to automatic)
        */      
      AdaptiveParticleFilterOMPTracker (unsigned int nr_threads = 0)
      : KLDAdaptiveParticleFilterTracker<PointInT, StateT> ()
      , threads_ (nr_threads)
      , last_stamp_ (0)
      {
        tracker_name_ = "AdaptiveParticleFilterOMPTracker";
        T1 = T2 = T3 = 0;
        frame_count = 0;
        changed_ = true;
      }

      /** \brief Initialize the scheduler and set the number of threads to use.
        * \param nr_threads the number of hardware threads to use (0 sets the value back to automatic)
        */
      inline void
      setNumberOfThreads (unsigned int nr_threads = 0) { threads_ = nr_threads; }


      
      virtual void 
      setInputCloud (const CloudInTConstPtr &cloud, double t_stamp = 0.0);
        
      void printParticleWeights ()
      {
        for (int i = 0; i < particle_num_; ++i)
          std::cout << particles_->points[i].weight<<"\n";
      }
      
      inline void
      setRejectiveCoherence (const RejCoherencePtr &coherence) { rejective_coherence_ = coherence; }
      
      inline RejCoherencePtr
      getRejCoherence () const { return rejective_coherence_; }
      
      bool 
      initCompute ();
      
    protected:
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::bin_size_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::insertIntoBins;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::maximum_particle_number_;
      using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::calcKLBound;
      
 
      
      /** \brief The number of threads the scheduler should use. */
      unsigned int threads_;

      /** \brief weighting phase of particle filter method.
          calculate the likelihood of all of the particles and set the weights.
        */
      virtual void weight ();
      
      void 
      update ();

      virtual void 
      resample ();

      double T1, T2, T3;
      int frame_count;
      double last_stamp_, delta_t_;

      RejCoherencePtr rejective_coherence_;
      
    };
  }
}
#include <tracker/impl/adaptive_particle_filter_omp.hpp>

#endif

