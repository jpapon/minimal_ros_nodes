#ifndef PCL_TRACKING_REJECTIVE_POINT_CLOUD_COHERENCE_H_
#define PCL_TRACKING_REJECTIVE_POINT_CLOUD_COHERENCE_H_

#include <pcl/search/kdtree.h>

#include <pcl/tracking/coherence.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

namespace pcl
{
  namespace tracking
  {
    /** \brief @b RejectivePointCloudCoherence 
     */
    template <typename PointInT>
    class RejectivePointCloudCoherence: public PointCloudCoherence<PointInT>
    {
    public:
      using PointCloudCoherence<PointInT>::getClassName;
      using PointCloudCoherence<PointInT>::coherence_name_;
      using PointCloudCoherence<PointInT>::target_input_;
      
      typedef typename PointCloudCoherence<PointInT>::PointCoherencePtr PointCoherencePtr;
      typedef typename PointCloudCoherence<PointInT>::PointCloudInConstPtr PointCloudInConstPtr;
      typedef PointCloudCoherence<PointInT> BaseClass;
      
      typedef boost::shared_ptr<RejectivePointCloudCoherence<PointInT> > Ptr;
      typedef boost::shared_ptr<const RejectivePointCloudCoherence<PointInT> > ConstPtr;
      typedef boost::shared_ptr<pcl::search::KdTree<PointInT> > SearchPtr;
      typedef boost::shared_ptr<const pcl::search::Search<PointInT> > SearchConstPtr;
      
      /** \brief empty constructor */
      RejectivePointCloudCoherence ()
      : new_target_ (false)
      , search_ ()
      , maximum_distance_ (std::numeric_limits<double>::max ())
      , num_samples_ (0)
      , h_weight_ (10.0)
      , s_weight_ (1.0)
      , v_weight_ (1.0)
      {
        coherence_name_ = "RejectivePointCloudCoherence";
        TESTVAL = 0;
        
      }
      
      /** \brief Provide a pointer to a dataset to add additional information
       * to estimate the features for every point in the input dataset.  This
       * is optional, if this is not set, it will only use the data in the
       * input cloud to estimate the features.  This is useful when you only
       * need to compute the features for a downsampled cloud.  
       * \param cloud a pointer to a PointCloud message
       */
      inline void 
      setSearchMethod (const SearchConstPtr search) { search_ = search; }
      
      /** \brief Get a pointer to the point cloud dataset. */
      inline SearchConstPtr 
      getSearchMethod () { return (search_); }
      
      /** \brief add a PointCoherence to the PointCloudCoherence.
       * \param[in] cloud coherence a pointer to PointCoherence.
       */
      virtual inline void
      setTargetCloud (const PointCloudInConstPtr cloud)
      {
        new_target_ = true;
        target_input_ = cloud;
      }
      
      void 
      setColorWeights (float h, float s, float v)
      {
        h_weight_ = h; s_weight_ = s; v_weight_ = v;
      }
      
      void
      setSamplePoints (int samples) { num_samples_ = samples;}
      
      /** \brief set maximum distance to be taken into account.
       * \param[in] val maximum distance.
       */
      inline void setMaximumDistance (double val) { maximum_distance_ = val; max_dist_squared_ = val*val;}
      
      
      /** \brief compute the nearest pairs and compute coherence using point_coherences_ */
      void
      computeSampledCoherence (const PointCloudInConstPtr &cloud, const Eigen::Affine3f &trans, float &w_j);
      
      
    protected:
      using PointCloudCoherence<PointInT>::point_coherences_;
      
      double
      computeScore (const PointInT& p1, const PointInT& p2, float dist = 0);
      
      
      double
      computeScoreHSV (const PointInT& p1, const PointInT& p2, float dist = 0);
      
      /** \brief This method should get called before starting the actual computation. */
      virtual bool initCompute ();
      
      /** \brief A flag which is true if target_input_ is updated */
      bool new_target_;
      
      /** \brief A pointer to the spatial search object. */
      SearchConstPtr search_;
      
      /** \brief max of distance for points to be taken into account*/
      double maximum_distance_, max_dist_squared_;

      virtual void
      computeCoherence (const PointCloudInConstPtr &cloud, const IndicesConstPtr &indices, float &w_j) 
      {
        std::cout << "CALLING VIRTUAL COMPUTE COHERENCE BAD BAD BAD \n";
      }
      
      int num_samples_;
      int TESTVAL;
      boost::mt19937 rng_;
      float h_weight_, s_weight_, v_weight_;
    };
  }
}


#include <tracker/impl/rejective_point_cloud_coherence.hpp>


#endif
