#ifndef MULTI_MODEL_TRACKER_H_
#define MULTI_MODEL_TRACKER_H_

#include <tracker/adaptive_particle_filter_omp.h>
#include <tracker/combined_coherence.h>
#include <tracker/rejective_point_cloud_coherence.h>

#include <pcl/tracking/tracking.h>
#include <pcl/tracking/tracker.h>
#include <pcl/tracking/coherence.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/tracking/distance_coherence.h>
#include <pcl/tracking/normal_coherence.h>
#include <pcl/tracking/hsv_color_coherence.h>
#include <pcl/common/time.h>

#include <Eigen/Dense>

#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_estimation.h>


/** \brief @b  MultiModelTracker tracks the groups of models which are given by
  * addCloudToTrack within the input measurement cloud
  * \author Jeremie Papon
  * \ingroup tracking
  */
namespace pcl 
{
  namespace tracking
  {
    template <typename PointInT, typename StateT>
    class MultiModelTracker: public pcl::tracking::Tracker<PointInT, StateT>
    {
      friend class AdaptiveParticleFilterOMPTracker<PointInT, StateT>;
      protected:
        using Tracker<PointInT, StateT>::deinitCompute;
        
      public:
        using Tracker<PointInT, StateT>::tracker_name_;
        using Tracker<PointInT, StateT>::input_;
        using Tracker<PointInT, StateT>::indices_;
        using Tracker<PointInT, StateT>::getClassName;
        
        typedef Tracker<PointInT, StateT> BaseClass;
        
        typedef typename Tracker<PointInT, StateT>::PointCloudIn PointCloudIn;
        typedef typename PointCloudIn::Ptr PointCloudInPtr;
        typedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;

        typedef pcl::PointXYZRGBL LabelPointT;
        typedef pcl::PointCloud<LabelPointT> LabelCloudT;

        typedef typename Tracker<PointInT, StateT>::PointCloudState PointCloudState;
        typedef typename PointCloudState::Ptr PointCloudStatePtr;
        typedef typename PointCloudState::ConstPtr PointCloudStateConstPtr;

        typedef PointCoherence<PointInT> Coherence;
        typedef boost::shared_ptr< Coherence > CoherencePtr;
        typedef boost::shared_ptr< const Coherence > CoherenceConstPtr;

        typedef PointCloudCoherence<PointInT> CloudCoherence;
        typedef boost::shared_ptr< CloudCoherence > CloudCoherencePtr;
        typedef boost::shared_ptr< const CloudCoherence > CloudCoherenceConstPtr;
        
        typedef PointCloud<PointInT> CloudInT;
        typedef boost::shared_ptr< CloudInT > CloudInTPtr;
        typedef boost::shared_ptr< const CloudInT > CloudInTConstPtr;
        
        typedef pcl::tracking::AdaptiveParticleFilterOMPTracker<PointInT, pcl::tracking::ParticleXYZRPY> ParticleFilter;
        typedef boost::shared_ptr< ParticleFilter > ParticleFilterPtr;
        typedef boost::shared_ptr< const ParticleFilter > ParticleFilterConstPtr;
        
        typedef std::pair<int,double> IndexDistPair;
        typedef std::vector<IndexDistPair> Point2ResultsDists;
        typedef std::vector<Point2ResultsDists> PointDistVec;
        typedef std::vector<std::set<uint32_t> > CandidateLabelVec;
        
        typedef pcl::octree::OctreePointCloudVoxelCentroid<PointInT> OctreeT;
        typedef boost::shared_ptr< OctreeT > OctreeTPtr;
        
        MultiModelTracker (float voxel_resolution);
        
        void
        addCloudToTrack (CloudInTConstPtr cloud_to_track, uint32_t label, bool adaptive_model = false);
        
        virtual void 
        setInputCloud (const CloudInTConstPtr &cloud, double t_stamp = 0.0);
        
        void
        setSamplePoints (int num_points) { sample_points_ = num_points; }
        
        void
        setNumParticles (int num_particles) { num_particles_ = num_particles; }
        
        void 
        getResultCloud (uint32_t label, LabelCloudT &result_cloud, uint32_t color = 0);
        
        void
        getColoredResultCloud (uint32_t label, PointCloudIn &result_cloud);
        
        void 
        getRefinedResultCloud (uint32_t label, LabelCloudT &refined_result_cloud, uint32_t color );
        
        /** \brief Get a pointer to a pointcloud of the particles. */
        PointCloudStatePtr 
        getParticles (uint32_t label) const;
        
        /** \brief This must be implemented due to inheritence - but does nothing 
          * since we have multiple things to track ie, multiple results
          */
        virtual inline StateT 
        getResult () const { return StateT(); }
        
        StateT
        getResult (uint32_t label) const { return particle_filters_.at(label)->getResult (); }
        
        StateT
        getRefinedResult (uint32_t label) const;
        
        StateT
        getRelativeResult (uint32_t label) const;
        
        CloudInTPtr
        getModelCloud (uint32_t label) { return model_clouds_[label];}
        
        void
        setThreadCounts (int total_threads);
        
        /** \brief Track the clouds using the particle filters. */
        bool  
        doTracking ();
        
        virtual void
        computeTracking () { bool result = doTracking (); }
        
        void 
        setColorWeights (float h, float s, float v)
        {
          h_weight_ = h; s_weight_ = s; v_weight_ = v;
        }
      protected:

        
        void
        createTransformedResults ();
        
        void
        refineResults ();
        
        void 
        updateModelCloud (uint32_t label, pcl::CorrespondencesConstPtr points_to_add);
        
        void 
        alignCorrespondencesToModels ();
        
      private:
        std::map <uint32_t, ParticleFilterPtr> particle_filters_;
        std::vector<uint32_t> id_vector_;
        std::map <uint32_t, CloudInTPtr> result_clouds_;
        std::map <uint32_t, CloudInTPtr> refined_result_clouds_;
        std::map <uint32_t, CloudInTPtr> visible_result_clouds_;
        std::map <uint32_t, CloudInTPtr> complete_result_clouds_;
        std::map <uint32_t,typename pcl::tracking::RejectivePointCloudCoherence<PointInT>::Ptr > coherences_;
        
        std::map <uint32_t, CloudInTPtr> model_clouds_;

        std::map <uint32_t, bool> variable_model_;
        std::map <uint32_t, Eigen::Affine3f> aligned_transform_map_;
        std::map <uint32_t, StateT> refined_state_;
        
        std::map <uint32_t, StateT> frozen_state_;
        std::map <uint32_t, int> model_frames_touching_;
        int touching_threshold_, min_points_touching_, point_density_;
        double touching_distance_;
        
        std::vector<double> step_covariance_,step_covariance_touching_;
        std::vector<double> initial_noise_covariance_;
        std::vector<double> initial_mean_;
      
        typename pcl::tracking::RejectivePointCloudCoherence<PointInT>::Ptr coherence_;
        boost::shared_ptr<pcl::tracking::DistanceCoherence<PointInT> > distance_coherence_;
        boost::shared_ptr<pcl::tracking::NormalCoherence<PointInT> > normal_coherence_;
        boost::shared_ptr<pcl::tracking::HSVColorCoherence<PointInT> > color_coherence_;
        
        pcl::CorrespondencesPtr correspondences_;
        pcl::CorrespondencesPtr correspondences_rej_SAC_;
        pcl::CorrespondencesPtr correspondences_final_;
        
        LabelCloudT::Ptr  result_cloud_;
        int num_particles_;
        int sample_points_;
        float voxel_resolution_;
        int frame_count_;
        double T1;
        int total_size_;
        double last_stamp_, start_stamp_, delta_stamp_;
        boost::shared_ptr<pcl::search::KdTree<PointInT> > kdtree_;
        
        boost::shared_ptr<pcl::search::KdTree<LabelPointT> > touch_kdtree_;
        
        float h_weight_,s_weight_,v_weight_;
    };
  }
}

#include <impl/multi_model_tracker.hpp>

#endif //MULTI_MODEL_TRACKER_H_
