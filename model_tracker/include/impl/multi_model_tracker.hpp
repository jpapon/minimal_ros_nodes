#ifndef MULTI_MODEL_TRACKER_HPP_
#define MULTI_MODEL_TRACKER_HPP_

#include "multi_model_tracker.h"

#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>

#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
//#include <pcl/registration/correspondence_estimation_backprojection.h>
//#include <pcl/registration/correspondence_estimation_joint_ransac.h>

#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>

#include <pcl/features/fpfh.h>
#include <pcl/registration/icp.h>
#include <sys/stat.h>

template <typename PointInT, typename StateT>
pcl::tracking::MultiModelTracker<PointInT, StateT>::MultiModelTracker (float voxel_resolution)
  :num_particles_(200)
  ,voxel_resolution_ (voxel_resolution)
  ,frame_count_(0)
  ,touching_threshold_(8)
  ,min_points_touching_(5)
  ,touching_distance_(0.04)
  ,total_size_(0)
  ,point_density_(10)
  ,last_stamp_(0)
  ,start_stamp_(0)
  ,delta_stamp_(0)
  ,sample_points_ (150)
  ,h_weight_(10)
  ,s_weight_(2)
  ,v_weight_(2)
{
  step_covariance_ = std::vector<double> (6, 0.04 * 0.01);
  
  step_covariance_[3] *= 25.0;
  step_covariance_[4] *= 25.0;
  step_covariance_[5] *= 25.0;
  initial_noise_covariance_ = std::vector<double> (6, 0.00001);
  initial_mean_ = std::vector<double> (6, 0.0);
  // Step covariance for touching objects
  step_covariance_touching_ = std::vector<double> (6, 0.005 * 0.001);
  step_covariance_touching_[3] *= 25.0;
  step_covariance_touching_[4] *= 25.0;
  step_covariance_touching_[5] *= 25.0;
  
  correspondences_ = boost::make_shared <pcl::Correspondences> ();
  correspondences_rej_SAC_ = boost::make_shared <pcl::Correspondences> ();
  correspondences_final_ = boost::make_shared <pcl::Correspondences> ();
  result_cloud_ = boost::make_shared<LabelCloudT>(); 
  T1 = 0;
  
  touch_kdtree_ = boost::make_shared<pcl::search::KdTree<LabelPointT> > (false);
  kdtree_ = boost::make_shared<pcl::search::KdTree<PointInT> > (false);
}

template <typename PointInT, typename StateT> void
pcl::tracking::MultiModelTracker<PointInT, StateT>::addCloudToTrack (CloudInTConstPtr cloud_to_track, uint32_t label, bool adaptive_model)
{
  ParticleFilterPtr tracker = boost::make_shared<ParticleFilter>(1);
  tracker->setTrans (Eigen::Affine3f::Identity ());
  tracker->setStepNoiseCovariance (step_covariance_);
  tracker->setInitialNoiseCovariance (initial_noise_covariance_);
  tracker->setInitialNoiseMean (initial_mean_);
  tracker->setIterationNum (1);
  tracker->setResampleLikelihoodThr(0);
  tracker->setUseNormal (false); 
  tracker->setMaximumParticleNum (num_particles_);
  tracker->setParticleNum (num_particles_);
  tracker->setMotionRatio (0.75);
  tracker->setAlpha (15.0);
  
  boost::shared_ptr<pcl::tracking::RejectivePointCloudCoherence<PointInT> > coherence = boost::make_shared<pcl::tracking::RejectivePointCloudCoherence<PointInT> >();
  coherence->setSamplePoints (sample_points_);
  coherence->setSearchMethod (kdtree_);
  coherence->setMaximumDistance (voxel_resolution_*1.0f);
  coherence->setColorWeights (h_weight_,s_weight_,v_weight_);
  tracker->setRejectiveCoherence (coherence);
  
  Eigen::Vector4f c;
  CloudInTPtr transed_ref = boost::make_shared<CloudInT> ();
  pcl::compute3DCentroid<PointInT> (*cloud_to_track, c);
  Eigen::Affine3f trans = Eigen::Affine3f::Identity ();
  trans.translation ().matrix () = Eigen::Vector3f (c[0], c[1], c[2]);
  pcl::transformPointCloud<PointInT> (*cloud_to_track, *transed_ref, trans.inverse());
  tracker->setReferenceCloud (transed_ref);
  tracker->setTrans (trans);
  tracker->setMinIndices (int (cloud_to_track->points.size ()) / 2);
  
  tracker->setMaximumParticleNum (num_particles_);
  tracker->setDelta (0.99);
  tracker->setEpsilon (0.1);
  StateT bin_size;
  bin_size.x = 0.01f;
  bin_size.y = 0.01f;
  bin_size.z = 0.01f;
  bin_size.roll = 0.05f;
  bin_size.pitch = 0.05f;
  bin_size.yaw = 0.05f;
  tracker->setBinSize (bin_size);

  particle_filters_[label] = tracker;
  id_vector_.push_back (label);
  //Make a local copy of the cloud to track as a model we can update
  model_clouds_[label] = boost::make_shared<CloudInT> ();
  pcl::copyPointCloud (*transed_ref, *model_clouds_[label]);
  //Initialize result pointer 
  result_clouds_[label] = boost::make_shared<CloudInT> ();
  refined_result_clouds_[label] = boost::make_shared<CloudInT> ();
  visible_result_clouds_[label] = boost::make_shared<CloudInT> ();
  complete_result_clouds_[label] = boost::make_shared<CloudInT> ();
  coherences_[label] = coherence;

  variable_model_[label] = adaptive_model;
  model_frames_touching_[label] = 5;
  total_size_ += transed_ref->size ();
  result_cloud_->resize (total_size_/point_density_);
  
}

template <typename PointInT, typename StateT> void
pcl::tracking::MultiModelTracker<PointInT, StateT>::setThreadCounts (int total_threads)
{
  int total_pts = 0;
  typename std::map <uint32_t, CloudInTPtr>::iterator model_itr = model_clouds_.begin ();
  for ( ; model_itr != model_clouds_.end (); ++model_itr)
  {
    total_pts += model_itr->second->size (); 
  }
  std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
  typename std::map <uint32_t, ParticleFilterPtr>::iterator pf_itr = particle_filters_.begin ();
  model_itr = model_clouds_.begin ();
  for ( ;  pf_itr != particle_filters_.end (); ++pf_itr, ++model_itr)
  {
    //int num_threads =  total_threads * static_cast<float> (model_itr->second->size ()) / total_pts;
    int num_threads =  total_threads  / particle_filters_.size ();
    
    num_threads = (num_threads < 1) ? 2 : num_threads + 1;
      
    pf_itr->second->setNumberOfThreads (num_threads);
    std::cout << "Set thread count for label of size "<<model_itr->second->size ()<< "  to " << num_threads<<std::endl;
  }
  std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";

}

template <typename PointInT, typename StateT> void
pcl::tracking::MultiModelTracker<PointInT, StateT>::setInputCloud (const CloudInTConstPtr &cloud, double t_stamp)
{
  if (start_stamp_ == 0)
    start_stamp_ = last_stamp_ = t_stamp;
  delta_stamp_ = t_stamp - last_stamp_; 
  last_stamp_ = t_stamp;
  if (delta_stamp_ == 0 && last_stamp_ != 0)
  {
  //  std::cout << "!!!!!!!!!!!!!!   Set Input cloud with same cloud! Doing nothing   !!!!!!!!!!!!!\n";
    return;
  }
  if (cloud == 0 || cloud->points.size () == 0)
  {
   // std::cout << " !!!!!!!!!!!! Blank cloud in set input, doing nothing            !!!!!!!\n";
    return;
  }
  Tracker<PointInT, StateT>::setInputCloud (cloud);
  
  kdtree_->setInputCloud (input_);
   
  typename std::map <uint32_t, ParticleFilterPtr>::iterator filter_itr = particle_filters_.begin ();
  for (; filter_itr != particle_filters_.end (); ++filter_itr)
  {
    (filter_itr->second)->setInputCloud (input_, t_stamp);
    (filter_itr->second)->getRejCoherence ()->setTargetCloud (input_);
    
  }
  
}

template <typename PointInT, typename StateT> bool
pcl::tracking::MultiModelTracker<PointInT, StateT>::doTracking ()
{
  if (delta_stamp_ == 0 && last_stamp_ != 0) //If cloud is same as last cloud, no need to compute
  {
   // std::cout << "!!!!!!!!!!!!!!   Tracker new frame same as old frame - doing nothing   !!!!!!!!!!!!!\n";
    return false;
  }
  frame_count_++;
  
  //Compute the tracking for each filter
  #pragma omp parallel for schedule(static, 1) num_threads( id_vector_.size () )
  for (int i = 0; i < id_vector_.size (); ++i)
  {
    int label = id_vector_[i];
    particle_filters_.at(label)->compute ();   
    if(frame_count_ == 1)
    {
      refined_state_ [label] =particle_filters_.at(label)->getResult ();      
    }
  } 
  createTransformedResults ();
  return true;
}

template <typename PointInT, typename StateT> void
pcl::tracking::MultiModelTracker<PointInT, StateT>::createTransformedResults ()
{
  //For each tracked cloud, create the best guess result
  typename std::map <uint32_t, CloudInTPtr>::iterator result_itr = result_clouds_.begin ();
  for (; result_itr != result_clouds_.end (); ++result_itr)
  {
    ParticleFilterPtr filter = particle_filters_.at (result_itr->first);
    ParticleXYZRPY result = filter->getResult ();
    Eigen::Affine3f transformation = filter->toEigenMatrix (result);
    pcl::transformPointCloud<PointInT> (*(filter->getReferenceCloud ()), *(result_itr->second), transformation);
  }
}


template <typename PointInT, typename StateT> void
pcl::tracking::MultiModelTracker<PointInT, StateT>::refineResults ()
{
  typename std::map <uint32_t, CloudInTPtr>::iterator result_itr = result_clouds_.begin ();
  typename std::map <uint32_t, CloudInTPtr>::iterator refined_result_itr = refined_result_clouds_.begin ();
  for (; result_itr != result_clouds_.end (); ++result_itr, ++refined_result_itr)
  {
    ParticleFilterPtr filter = particle_filters_.at (result_itr->first);
    ParticleXYZRPY filter_result = filter->getResult ();
    refined_state_[result_itr->first] =  filter_result;
    pcl::transformPointCloud<PointInT> (*(filter->getReferenceCloud ()), *(refined_result_itr->second),refined_state_[result_itr->first].toEigenMatrix ());
  }
}

template <typename PointInT, typename StateT> void
pcl::tracking::MultiModelTracker<PointInT, StateT>::updateModelCloud (uint32_t label, pcl::CorrespondencesConstPtr points_to_add)
{
  
  //Only add non-rejected points to the model
  pcl::PointIndices good_points;
  good_points.indices.reserve (points_to_add->size());
  for (int i = 0; i < points_to_add->size(); ++i)
    good_points.indices.push_back ( (*points_to_add)[i].index_query);
  CloudInT transformed_vis_result;
  //We need to update the model, while not deleting occluded supervoxels
  //This means we use the complete_result_clouds, rather than just visible result cloud
  CloudInTConstPtr complete_result = complete_result_clouds_.at(label);
  pcl::transformPointCloud (*complete_result, transformed_vis_result, aligned_transform_map_.at(label));
  //Set all label values and copy RGB values 
  for (int index = 0; index < transformed_vis_result.size (); ++index)
  {
    transformed_vis_result.points[index].label = label;
    transformed_vis_result.points[index].R = complete_result->points[index].R;
    transformed_vis_result.points[index].G = complete_result->points[index].G;
    transformed_vis_result.points[index].B = complete_result->points[index].B;
    transformed_vis_result.points[index].r = complete_result->points[index].r;
    transformed_vis_result.points[index].g = complete_result->points[index].g;
    transformed_vis_result.points[index].b = complete_result->points[index].b;
  }
  
  //Update the model cloud
  model_clouds_[label] = transformed_vis_result.makeShared ();
  
}

template <typename PointInT, typename StateT> void
pcl::tracking::MultiModelTracker<PointInT, StateT>::alignCorrespondencesToModels ()
{
  //Now we need to find correspondence from matched supervoxels into model so we can estimate transforms
  typename std::map<uint32_t, CloudInTPtr>::iterator model_itr = model_clouds_.begin ();
  typename std::map<uint32_t, CloudInTPtr >::iterator vis_result_itr = visible_result_clouds_.begin ();
  typename std::map <uint32_t, ParticleFilterPtr>::iterator pf_itr = particle_filters_.begin ();
  
  pcl::registration::TransformationEstimationSVD<PointInT,PointInT> transform_est;
  //Iterate through, generating a transform for each set of supervoxels to the model
  pcl::registration::CorrespondenceEstimation<PointInT,PointInT> est;
  pcl::registration::CorrespondenceRejectorSampleConsensus<PointInT> corrs_rejector_SAC;
  pcl::CorrespondencesPtr correspondences (new pcl::Correspondences);
  pcl::CorrespondencesPtr correspondences_rej_SAC (new pcl::Correspondences);
  
  CloudInTPtr trans_supervoxels = boost::make_shared<CloudInT> ();
  Eigen::Vector4f c;
  
  for ( ; model_itr != model_clouds_.end (); ++model_itr, ++vis_result_itr, ++pf_itr)
  {

    //Transform the new supervoxels to the origin as a rough guess to the alignment to model
    //Also use rotation from particle filter result 
    pcl::compute3DCentroid<PointInT> (*vis_result_itr->second, c);
    StateT result_state = pf_itr->second->getResult ();
    Eigen::Affine3f trans = pcl::getTransformation (c[0], c[1], c[2], 
                                                    result_state.roll, result_state.pitch, result_state.yaw);
    
    pcl::transformPointCloud<PointInT> (*vis_result_itr->second, *trans_supervoxels, trans.inverse());
    
    //Now find correspondences between new supervoxels associated with this label and its model
    est.setInputSource (trans_supervoxels);
    est.setInputTarget (model_itr->second);
    est.determineCorrespondences (*correspondences);
    //If we don't have enough visible correspondences, we can't correct PF or update model
    if (correspondences->size () < 4)
    {
      std::cout << "Not enough correspondences to align supervoxels to model "<<model_itr->first<<"\n";
      //So we clear the bin in the transform map
      aligned_transform_map_.erase (model_itr->first);
      continue;
    }
    //Now we do a RANSAC rejection - just for good measure
    corrs_rejector_SAC.setInputSource(trans_supervoxels);
    corrs_rejector_SAC.setInputTarget(model_itr->second);
    corrs_rejector_SAC.setInlierThreshold(0.03);
    corrs_rejector_SAC.setMaximumIterations(500);
    corrs_rejector_SAC.setInputCorrespondences(correspondences);
    corrs_rejector_SAC.getCorrespondences(*correspondences_rej_SAC);
    //Now estimate a transform for new supervoxels to model
   // std::cout << "Corr before RANSAC:"<<correspondences->size ()<<" corr after:"<<correspondences_rej_SAC->size ()<<"\n";
    //Still need 4 good correspondences...
    if (correspondences_rej_SAC->size () < 4)
    {
      std::cout << "Not enough correspondences to align supervoxels to model "<<model_itr->first<<"\n";
      //So we clear the bin in the transform map
      aligned_transform_map_.erase (model_itr->first);
      continue;
    }
    //TODO: ICP to refine transform?
    //Estimate a transform from global to model based on correspondences 
    typename pcl::registration::TransformationEstimationSVD<PointInT,PointInT>::Matrix4 transform;
    transform_est.estimateRigidTransformation (*vis_result_itr->second, *model_itr->second, *correspondences_rej_SAC, transform);
    
    aligned_transform_map_[model_itr->first] = (Eigen::Affine3f (transform));//.inverse ();
    
    StateT est_state, refined_state;
    pcl::getTranslationAndEulerAngles (trans, est_state.x, est_state.y, est_state.z, est_state.roll, est_state.pitch, est_state.yaw);
    pcl::getTranslationAndEulerAngles (aligned_transform_map_[model_itr->first].inverse (), refined_state.x, refined_state.y, refined_state.z, refined_state.roll, refined_state.pitch, refined_state.yaw);
    
    particle_filters_[model_itr->first]->refineStates (refined_state);
 //   std::cout << "Estimated state:"<<est_state<<"\n";
 //   std::cout << "Predicted state:"<<result_state<<"\n";
 //   std::cout << "Refined state:"<<refined_state<<"\n";
    
    if ( variable_model_[model_itr->first] )
      updateModelCloud (model_itr->first, correspondences_rej_SAC);
    
  }
  
}

template <typename PointInT, typename StateT> StateT
pcl::tracking::MultiModelTracker<PointInT, StateT>::getRefinedResult (uint32_t label) const
{
  return refined_state_.at (label);
}

template <typename PointInT, typename StateT> void
pcl::tracking::MultiModelTracker<PointInT, StateT>::getResultCloud (uint32_t label, LabelCloudT &result_cloud, uint32_t color )
{
  copyPointCloud ( *(result_clouds_.at (label)) ,  result_cloud);
  for (LabelCloudT::iterator itr = result_cloud.begin(); itr != result_cloud.end (); ++itr)
  {
    itr->label = label;
    itr->rgb = color;
  }
}

template <typename PointInT, typename StateT> void
pcl::tracking::MultiModelTracker<PointInT, StateT>::getColoredResultCloud (uint32_t label, PointCloudIn &result_cloud)
{
  copyPointCloud ( *(result_clouds_.at (label)) ,  result_cloud);
}

template <typename PointInT, typename StateT> void
pcl::tracking::MultiModelTracker<PointInT, StateT>::getRefinedResultCloud (uint32_t label, LabelCloudT &refined_result_cloud, uint32_t color )
{
  copyPointCloud ( *(refined_result_clouds_.at (label)) ,  refined_result_cloud);
  for (LabelCloudT::iterator itr = refined_result_cloud.begin(); itr != refined_result_cloud.end (); ++itr)
  {
    itr->label = label;
    itr->rgb = color;
  }
}



template <typename PointInT, typename StateT> typename pcl::tracking::MultiModelTracker<PointInT, StateT>::PointCloudStatePtr
pcl::tracking::MultiModelTracker<PointInT, StateT>::getParticles (uint32_t label) const
{
  return (particle_filters_.at (label))->getParticles (); 
  
}

template <typename PointInT, typename StateT> StateT
pcl::tracking::MultiModelTracker<PointInT, StateT>::getRelativeResult (uint32_t label) const
{
 
  StateT current_result = (particle_filters_.at (label))->getResult ();
  Eigen::Vector3f translation = (particle_filters_.at (label))->getTrans ().translation ();
  current_result.x -= translation [0]; 
  current_result.y -= translation [1];
  current_result.z -= translation [2];

  return current_result;
  
}

#endif
