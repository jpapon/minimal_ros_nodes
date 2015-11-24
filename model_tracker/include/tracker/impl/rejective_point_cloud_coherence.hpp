#ifndef PCL_TRACKING_IMPL_REJECTIVE_POINT_CLOUD_COHERENCE_H_
#define PCL_TRACKING_IMPL_REJECTIVE_POINT_CLOUD_COHERENCE_H_

#include <pcl/search/kdtree.h>
#include <pcl/search/organized.h>
#include <tracker/rejective_point_cloud_coherence.h>
#include <tracker/combined_coherence.h>

namespace pcl
{
  namespace tracking
  {
    template <typename PointInT> void 
    RejectivePointCloudCoherence<PointInT>::computeSampledCoherence (
      const PointCloudInConstPtr &cloud, const Eigen::Affine3f &trans, float &w)
    {
   //   std::unordered_multimap<int, int> index_map;
   //   index_map.reserve (cloud->size ());
   //   std::map <int,double> min_w_map;  
      std::map <int, int> target_model_map;
      //Compute Half of Hausdorff distance
      std::vector<int> k_indices(1);
      std::vector<float> k_distances(1);
      float max_dist = -1.0f * std::numeric_limits<float>::max ();
      float sum_max_dist = 0.0f;
      double val = 1.0;
      int interval = cloud->points.size () / num_samples_;
      PointInT transformed_pt; 
      //Now search for transformed_pt instead
      //Main problem - need to pass transform 
      //float r1=0,r2=0, g1=0,g2=0,b1=0,b2=0;
      if (interval < 2) //Special case - use every point, no random
      {
        for (size_t i = 0; i < cloud->points.size (); i++)
        {       
          transformed_pt.getVector3fMap () = trans * cloud->points[i].getVector3fMap ();
          transformed_pt.rgb = cloud->points[i].rgb;
          search_->nearestKSearch (transformed_pt, 1, k_indices, k_distances);
          k_distances[0] = sqrt (k_distances[0]);
          if (k_distances[0] > max_dist)
            max_dist = k_distances[0];
          if (k_distances[0] < maximum_distance_)
            val += computeScoreHSV (target_input_->points[k_indices[0]], transformed_pt, k_distances[0]);
          sum_max_dist += k_distances[0];
        }
      }
      else //otherwise randomly sample at regular intervals
      {
        boost::uniform_int<int> index_dist(0, interval);
        int max_offset = cloud->points.size () - interval;
        for (size_t offset = 0; offset < max_offset; offset += interval)
        {       
          int idx = offset + index_dist(rng_);
          transformed_pt.getVector3fMap () = trans * cloud->points[idx].getVector3fMap ();
          transformed_pt.rgb = cloud->points[idx].rgb;
          search_->nearestKSearch (transformed_pt, 1, k_indices, k_distances);
          k_distances[0] = sqrt (k_distances[0]);
          if (k_distances[0] > max_dist)
            max_dist = k_distances[0];
          if (k_distances[0] < maximum_distance_)
            val += computeScoreHSV (target_input_->points[k_indices[0]], transformed_pt, k_distances[0]);
          sum_max_dist += k_distances[0];
        }
      }
      
      //Max distance allowed - coherence is zero if ANY points are too far away
      if (max_dist > 0.05f)
      {
      //  std::cout <<"REJECTING MAX\n";
      w = -1.0f * std::numeric_limits<float>::min ();
        return;
      }
      //Avereage max distance of points - coherence is zero if average is too large
      float avg_max_dist = sqrt(sum_max_dist) / cloud->points.size ();
      if (avg_max_dist > 0.02f)
      {
       // std::cout <<"REJECTING AVG\n";
       w = -1.0f * std::numeric_limits<float>::min ();
        return;
      }
      w = - static_cast<float> (val);
    }
    
    template <typename PointInT> double 
    RejectivePointCloudCoherence<PointInT>::computeScore (const PointInT &p1,const PointInT& p2, float dist)
    {
      
      Eigen::Vector3f rgb1(p1.r, p1.g, p1.b);
      Eigen::Vector3f rgb2(p2.r, p2.g, p2.b);
      double d_rgb = (rgb1 - rgb2).norm () / 255.0f;
     // double d_spatial =  (p1.getVector4fMap () - p2.getVector4fMap ()).norm ();
      
      return (1.0 / (1.0 + d_rgb + dist));      
    }
      
    template <typename PointInT> double 
    RejectivePointCloudCoherence<PointInT>::computeScoreHSV (const PointInT &p1,const PointInT& p2, float dist)
    {
        
      // convert color space from RGB to HSV
      RGBValue source_rgb, target_rgb;
      source_rgb.int_value = p1.rgba;
      target_rgb.int_value = p2.rgba;
      
      float source_h, source_s, source_v, target_h, target_s, target_v;
      RGB2HSV (source_rgb.Red, source_rgb.Blue, source_rgb.Green,
               source_h, source_s, source_v);
      RGB2HSV (target_rgb.Red, target_rgb.Blue, target_rgb.Green,
               target_h, target_s, target_v);
      // hue value is in 0 ~ 2pi, but circulated.
      const float _h_diff = fabsf (source_h - target_h);
      // Also need to compute distance other way around circle - but need to check which is closer to 0
      float _h_diff2;
      if (source_h < target_h)
        _h_diff2 = fabsf (1.0f + source_h - target_h); //Add 2pi to source, subtract target
      else 
        _h_diff2 = fabsf (1.0f + target_h - source_h); //Add 2pi to target, subtract source
          
      float h_diff;
      //Now we need to choose the smaller distance
      if (_h_diff < _h_diff2)
        h_diff = static_cast<float> (h_weight_) * _h_diff * _h_diff;
      else
        h_diff = static_cast<float> (h_weight_) * _h_diff2 * _h_diff2;
      
      const float s_diff = static_cast<float> (s_weight_) * (source_s - target_s) * (source_s - target_s);
      const float v_diff = static_cast<float> (v_weight_) * (source_v - target_v) * (source_v - target_v);
      
      //const float color_diff = h_diff + s_diff + v_diff;
      //if (color_diff > 0.1)
      //  return 0;
      
      const float diff2 = h_diff + s_diff + v_diff + dist / maximum_distance_;
      
      return (1.0 / (1.0 + diff2));
    }
      
    template <typename PointInT> bool
    RejectivePointCloudCoherence<PointInT>::initCompute ()
    {
    /*  
      if (!PointCloudCoherence<PointInT>::initCompute ())
      {
        PCL_ERROR ("[pcl::%s::initCompute] RejectivePointCloudCoherence::Init failed.\n", getClassName ().c_str ());
        //deinitCompute ();
        return (false);
      }
      
      // initialize tree
      if (!search_)
        search_.reset (new pcl::search::KdTree<PointInT> (false));
      
      if (new_target_ && target_input_)
      {
        std::cout <<"SETTING INPUT CLOUD FOR SEARCH\n";
        search_->setInputCloud (target_input_);
        new_target_ = false;
      }
      */
      return true;
    }
  }
  
  
  
}



#define PCL_INSTANTIATE_RejectivePointCloudCoherence(T) template class PCL_EXPORTS pcl::tracking::RejectivePointCloudCoherence<T>;

#endif
