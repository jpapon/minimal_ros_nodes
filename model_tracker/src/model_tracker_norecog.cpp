// ROS
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

//PCLROS
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

// PCL
#include <pcl/common/time.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>

// BOOST
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/filesystem.hpp>

#include <sstream>
#include <fstream> 

//LOCAL
#include <multi_model_tracker.h>
#include <tracker/adaptive_particle_filter_omp.h>
#include <tracker/combined_coherence.h>
#include <tracker/rejective_point_cloud_coherence.h>

//Output Message Type
#include <model_tracker/ModelTracker.h>
typedef model_tracker::ModelTracker OutMsgT;

// Typedefs
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> CloudT;
typedef pcl::PointXYZRGB InputPointT;
typedef pcl::PointCloud<InputPointT> InputCloudT;
typedef pcl::PointXYZRGBL LabelPointT;
typedef pcl::PointCloud<LabelPointT> LabelCloudT;
typedef pcl::tracking::ParticleXYZRPY PosePointT;
typedef pcl::PointCloud<PosePointT> PosePointCloudT;
typedef sensor_msgs::PointCloud2 MsgCloudT;
typedef std::map<std::string,MsgCloudT::ConstPtr>  ConstMsgCloudMapT;
typedef std::map<std::string,CloudT::Ptr>  CloudMapT;
typedef std::map<std::string,Eigen::Matrix4f> TransformMapT;
typedef std::map<std::string, pcl::PointXYZ> CameraMapT;

template class pcl::tracking::AdaptiveParticleFilterOMPTracker<PointT, PosePointT>;
template class pcl::tracking::CombinedCoherence<PointT>;
template class pcl::tracking::RejectivePointCloudCoherence<PointT>;


//Global vars
bool show_no_ground_ = false;
bool show_voxels_ = true;
bool show_models_ = false;
bool show_result_ = true;
bool show_result_colored_ = false;
bool show_tracks_ = false;
bool show_names_ = false;
bool show_help_ = true;
bool show_particles_ = false;

MsgCloudT::ConstPtr last_cloud_;
boost::mutex mcloud;
std::map<uint32_t, uint32_t> label_remapping_;
std::map<std::string, int> camera_num_map_;
bool write_to_file_,use_touching_relation_;
int number_of_threads_;
double h_weight_,s_weight_,v_weight_;
double voxel_resolution;

// Global Pose Variables
typedef std::map<int,std::string> IDLabelMap;
typedef std::map<int,boost::array<double,16> > IDPoseMap;
typedef std::map<int,Eigen::Affine3f> IDTransMap;
IDLabelMap ID_to_label_;
IDPoseMap ID_to_pose_;
IDTransMap ID_to_conversion_;
std::map<uint32_t,uint32_t>  ID_to_color_;
double minX, minY, minZ;
double maxX, maxY, maxZ;

// Function declarations
void keyboardCallback (const pcl::visualization::KeyboardEvent& event, void*);
void clearParams (ros::NodeHandle&);
void pointcloudCallback (MsgCloudT::ConstPtr cloud_msg); 

void downsampleCloud (MsgCloudT::ConstPtr cloud_in, pcl::VoxelGrid<PointT> &voxel_grid, CloudT::Ptr &output);
void cropCloud (CloudT::ConstPtr input_cloud, pcl::CropBox<PointT>& crop_box_filter,CloudT::Ptr &output);
pcl::ModelCoefficients removeGroundPlane(CloudT::ConstPtr input_cloud, pcl::SACSegmentation<PointT>& seg, CloudT::Ptr &output);

void initializeTracker (const CloudT::ConstPtr combined_cloud,  
                        pcl::tracking::MultiModelTracker<PointT, PosePointT>& tracker,
                        const std::vector<pcl::PointIndices>& cluster_indices);

void initialSegmentation (const CloudT::ConstPtr combined_cloud, std::vector<pcl::PointIndices>& cluster_indices);

void printText (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer);
void removeText (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer);

//Whether or not to use the visualizer, mainly for debugging
#define DO_VISUALIZATION 1

// Main entry point
int main (int argc, char **argv) 
{
  // Init
  ros::init (argc, argv, "model_tracker_norecog");
   
  // Get node
  ros::NodeHandle n ("~");

	int frame_rate;
	double minX, minY, minZ;
	double maxX, maxY, maxZ;
	double ground_threshold;
  double min_delta;
	bool do_visualization, occlusion_testing;
  int num_samples, num_particles;
  
  std::string pointcloud_source_topic;
  // Get parameters
  n.param<std::string> ("source_topic", pointcloud_source_topic, "/sensors/cameras/masthead/binned/points2");
  n.param ("write_to_file",write_to_file_, false);
  n.param ("frame_rate", frame_rate, 1);
  n.param ("voxel_resolution", voxel_resolution, 0.01);
  n.param ("ground_threshold", ground_threshold, voxel_resolution*2.5);
  n.param ("visualization", do_visualization, true);
  n.param ("num_samples", num_samples, 100);
  n.param ("num_particles", num_particles, 200);
  n.param ("number_of_threads", number_of_threads_, 8);
  n.param ("hue_weight", h_weight_, 10.0);
  n.param ("saturation_weight", s_weight_, 2.0);
  n.param ("value_weight", v_weight_, 2.0);
    
  std::string items;
  n.param<std::string> ("items", items, "");
  std::set<std::string> items_to_track;
  boost::split (items_to_track, items, boost::is_any_of (",") );
  
  last_cloud_ = 0;
  ROS_INFO_STREAM ("Subscribing to pointcloud topic:  " << pointcloud_source_topic  << "\"...");
  ros::Subscriber sub_pointcloud = n.subscribe<MsgCloudT::ConstPtr> ( pointcloud_source_topic, 10, pointcloudCallback);
  // Advertise topics
  const std::string model_tracker_pub_name = "tracked_poses";
  ROS_INFO_STREAM ("Advertising topic \"" << model_tracker_pub_name << "\"...");
  ros::Publisher model_tracker_pub = n.advertise<model_tracker::ModelTracker> (model_tracker_pub_name, 1); 
  
  //Set up the objects we will use later for downsampling, segmentation etc...
  pcl::VoxelGrid<PointT> voxel_grid;
  voxel_grid.setLeafSize (voxel_resolution, voxel_resolution, voxel_resolution);
  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<PointT> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (2000);
  seg.setDistanceThreshold (ground_threshold);
  //Initialize tracker for tracking segmented objects
  pcl::tracking::MultiModelTracker<PointT, PosePointT> tracker(voxel_resolution);
  tracker.setSamplePoints (num_samples);
  tracker.setNumParticles (num_particles);
  tracker.setColorWeights (h_weight_,s_weight_,v_weight_);
  
  //Now we need to wait for the first pointcloud to arrive
  //Let's run this at frame_rate Hz
  ros::Rate r (frame_rate);
  ROS_ERROR_STREAM ("WAITING ON FIRST FRAME");
  while(ros::ok()) 
  {
    if (last_cloud_ != 0)
    {
      ROS_INFO_STREAM ("Received cloud of size "<< last_cloud_->width<<"x"<<last_cloud_->height);
      break;
    }
      
    ROS_INFO_THROTTLE(2,"Waiting on frame 0");
    r.sleep ();
    ros::spinOnce ();
  }
  
  MsgCloudT::ConstPtr initial_cloud = last_cloud_;

  
  
  // Now do the processing of the initial frame
  CloudT::Ptr downsampled_cloud (new CloudT);
  downsampleCloud (initial_cloud, voxel_grid, downsampled_cloud);  
  //Get the table-less clouds
  CloudT::Ptr no_ground_cloud (new CloudT);
  pcl::ModelCoefficients ground_coeffs = removeGroundPlane (downsampled_cloud, seg, no_ground_cloud);
  std::cout << "No ground plane cloud size = "<<no_ground_cloud->size ()<<"\n";
 
  //Initial segmentation into clusters
  std::vector<pcl::PointIndices> cluster_indices;
  initialSegmentation (no_ground_cloud, cluster_indices);
  ROS_INFO_STREAM ("Found "<<cluster_indices.size ()<< " segments");
  
  //Initialize tracker using the found clusters
  initializeTracker (no_ground_cloud, tracker, cluster_indices);
  
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  if (do_visualization)
  {
    ROS_ERROR_STREAM ("Creating tracker");
    viewer = boost::make_shared<pcl::visualization::PCLVisualizer> ("3D Viewer");
    viewer->setBackgroundColor (0, 0, 0);
    viewer->registerKeyboardCallback(keyboardCallback, 0);
    viewer->addPointCloud (downsampled_cloud, "voxels");
  //  viewer->setCameraPose(0, 0, -0.5, 0, 0, 1, 0, -1, 0);
    viewer->addCoordinateSystem (0.1);
    ROS_ERROR_STREAM ( "Done creating visualizer\n" );
  }
    
  LabelCloudT::Ptr tracker_output = boost::make_shared<LabelCloudT> ();  
  CloudT::Ptr colored_tracker_output = boost::make_shared<CloudT> ();
  LabelCloudT::Ptr combined_output = boost::make_shared<LabelCloudT> ();
  pcl::PointCloud <pcl::PointXYZI>::Ptr particle_cloud = boost::make_shared<pcl::PointCloud <pcl::PointXYZI> > ();
  LabelCloudT::Ptr tracks = boost::make_shared<LabelCloudT>();
  PosePointCloudT state_cloud; 
  
  pcl::StopWatch timer;
  int frame_count = 0, processed_frames = 0;
  int txt_out_period = 5; 
  double t_ds=0, t_comp=0, t_rest=0, t_tot = 0, t_viz = 0;
 
  double start = ros::Time::now().toSec (), last_now =ros::Time::now().toSec ();
  double prev_stamp = last_cloud_->header.stamp.toSec(), current_stamp;
  double first_stamp = last_cloud_->header.stamp.toSec();
  
  MsgCloudT::ConstPtr input_cloud;
  while(ros::ok()) 
  {
    bool tracked_frame = false;
    if (last_cloud_ != input_cloud)
    {
      input_cloud = last_cloud_;
      double t_start_total = timer.getTime ();
      downsampleCloud(input_cloud, voxel_grid, downsampled_cloud);  
      ground_coeffs = removeGroundPlane (downsampled_cloud, seg, no_ground_cloud);
    
      //Do tracking
      current_stamp = input_cloud->header.stamp.toSec();
      tracker.setInputCloud (no_ground_cloud, current_stamp);
      tracked_frame = tracker.doTracking ();

      //Get and publish the poses from the tracker
      ROS_INFO_STREAM_THROTTLE (txt_out_period, "================  Current Tracked Poses ==================");
      combined_output->clear ();
      OutMsgT output_message;
      output_message.poses.reserve (16*ID_to_label_.size ());
      output_message.object_strings.reserve (ID_to_label_.size ());
      output_message.labels.reserve (ID_to_label_.size ());
      state_cloud.reserve (ID_to_label_.size ());
      for (IDLabelMap::iterator ID_itr = ID_to_label_.begin (); ID_itr!=ID_to_label_.end (); ++ID_itr)
      {
        int ID = ID_itr->first;
        uint32_t rgb = ID_to_color_.at(ID_itr->first);
        pcl::tracking::ParticleXYZRPY current_state = tracker.getResult (ID);
        state_cloud.push_back (current_state);
        LabelPointT state_rgb;
        state_rgb.x = current_state.x; state_rgb.y = current_state.y; state_rgb.z = current_state.z; 
        state_rgb.rgb =rgb;
        state_rgb.label = ID;
        tracks->push_back (state_rgb);
        
        if (frame_count % txt_out_period == 0)
          ROS_INFO_STREAM (ID_to_label_.at(ID)<< "("<< ID << " state: "<< current_state);
        tracker.getResultCloud (ID, *tracker_output, rgb);
        *combined_output += *tracker_output;
        
        Eigen::Affine3f current_pose_tf = current_state.toEigenMatrix ();
        Eigen::Affine3f pose_current = current_pose_tf * ID_to_conversion_.at (ID);
        boost::array<double,16> pose_row_major;             //I hate you eigen
        for (int r=0,i=0; r<4; ++r)
          for (int c = 0; c<4; ++c, ++i)
            pose_row_major[i] = pose_current (r,c);
        output_message.poses.insert (output_message.poses.end(), pose_row_major.data(), pose_row_major.data()+16);
        output_message.object_strings.push_back (ID_to_label_.at (ID));
        output_message.labels.push_back (ID);
      }
      pcl::toROSMsg (state_cloud, output_message.states);
      pcl::toROSMsg (*tracks, output_message.tracks);
      pcl::toROSMsg (*combined_output, output_message.labeled_model_cloud);
      pcl::toROSMsg (*downsampled_cloud, output_message.raw_cloud);
      output_message.labeled_model_cloud.header.frame_id = input_cloud->header.frame_id;
      output_message.header.stamp = input_cloud->header.stamp;
      model_tracker_pub.publish (output_message);
      
      double t_end_total = timer.getTime ();
      double t_this_frame = t_end_total - t_start_total;
      if (t_this_frame > 1)
      {
        t_tot += t_this_frame;
        ++processed_frames;
      }

      ROS_INFO_STREAM_THROTTLE (txt_out_period,"=== Tracker done with frame "<<frame_count<<" t=" << t_this_frame <<"ms  (avg = "<<t_tot / processed_frames<<"ms) ===  ");
    }
    
    //The rest is all just visualization!
    //////////////////////////////////////////////////////////////////////////////////////////////
    double t_viz_start = timer.getTime (); 
    
    if (do_visualization)
    {
      particle_cloud->clear ();
      int k = 0;
      for (IDLabelMap::iterator ID_itr = ID_to_label_.begin (); ID_itr!=ID_to_label_.end (); ++ID_itr)
      {
        
        tracker.getColoredResultCloud (ID_itr->first, *colored_tracker_output);
        std::string name_colored = boost::str (boost::format ("result%06d_colored") % ID_itr->first );
        if (show_result_colored_)
        {
          pcl::visualization::PointCloudColorHandlerRGBField<PointT> result_color (colored_tracker_output);
          if (!viewer->updatePointCloud (colored_tracker_output, result_color, name_colored))
          {
            viewer->addPointCloud (colored_tracker_output, result_color, name_colored);
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5.0, name_colored);
          }
        }else
        {
          viewer->removePointCloud (name_colored);
        }
        
        pcl::tracking::ParticleXYZRPY current_state = tracker.getResult (ID_itr->first);
        std::string name_text = boost::str (boost::format ("result%06d_text") % ID_itr->first );
        if (show_names_)
        {
          std::ostringstream label_text;
          label_text << ID_to_label_.at(ID_itr->first) << "("<<ID_itr->first <<")";
          //std::string label_text = ID_to_label_.at(ID_itr->first);
          //if (!viewer->updateText (label_text, , , 10, 1.0, 1.0, 1.0, name_text) )
          viewer->removeShape (name_text);
          pcl::PointXYZ temp_point; 
          temp_point.x = current_state.x; temp_point.y = current_state.y; temp_point.z = current_state.z;
          temp_point.z -= 0.06f;
          viewer->addText3D (label_text.str(), temp_point,0.025 , 1.0, 1.0, 1.0, name_text);
        }
        else
        {
          viewer->removeShape (name_text);
        }
        
        if (show_particles_)
        {
          PosePointCloudT::Ptr particles = tracker.getParticles (ID_itr->first);
          pcl::PointCloud <pcl::PointXYZI>::Ptr temp_cloud (new pcl::PointCloud <pcl::PointXYZI>);
          pcl::copyPointCloud(*particles, *temp_cloud);
          for (int i = 0; i <particles->points.size (); ++i)
          {
            temp_cloud->points[i].intensity = particles->points[i].weight * 10; 
          }
          *particle_cloud+=*temp_cloud;
        } 
        
      }
      
      if (show_particles_)
      {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> particle_weights (particle_cloud,"intensity");
        if (! viewer->updatePointCloud (particle_cloud, particle_weights,"particles"))
        {
          viewer->addPointCloud (particle_cloud, particle_weights, "particles");
          viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "particles");
        }
      }else
      {
        viewer->removePointCloud ("particles");
      }
      
      
      if (show_result_)
      {
        pcl::visualization::PointCloudColorHandlerRGBField<LabelPointT> result_labeled (combined_output);
        if (!viewer->updatePointCloud (combined_output, result_labeled, "combined_output"))
        {
          viewer->addPointCloud (combined_output, result_labeled, "combined_output");
          viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4.0, "combined_output");
        }
      }else
      {
        viewer->removePointCloud ("combined_output");
      }
      
      if (show_tracks_)
      {
        pcl::visualization::PointCloudColorHandlerRGBField<LabelPointT> track_handler(tracks);
        if (!viewer->updatePointCloud (tracks, track_handler, "tracks"))
        {
          viewer->addPointCloud (tracks, track_handler, "tracks");
          viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,2.0, "tracks");
        }
      }else
      {
        viewer->removePointCloud ("tracks");
      }

      if (show_no_ground_)
      {
        
        if (!viewer->updatePointCloud (no_ground_cloud, "no_ground_cloud"))
        {
          viewer->addPointCloud (no_ground_cloud, "no_ground_cloud");
          viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,4.0, "no_ground_cloud");
        }
      }
      else
      {
        viewer->removePointCloud ("no_ground_cloud");
      }

      if (show_voxels_)
      {
        if (!viewer->updatePointCloud (downsampled_cloud, "voxels"))
        {
          viewer->addPointCloud (downsampled_cloud, "voxels");
          viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,4.0, "voxels");
        }
      }
      else
      {
        viewer->removePointCloud ("voxels");
      }
      
      if (show_help_)
      {
        viewer->removeShape ("help_text");
        printText (viewer);
      }
      else
      {
        removeText (viewer);
        if (!viewer->updateText("Press h to show help", 5, 10, 12, 1.0, 1.0, 1.0,"help_text") )
          viewer->addText("Press h to show help", 5, 10, 12, 1.0, 1.0, 1.0,"help_text");
      }
    }
    
    //If we actually processed a frame, save the results to file
    if (tracked_frame && write_to_file_) 
    {
      int framecounter = input_cloud->header.seq;
      int tstamp = input_cloud->header.stamp.toSec() * 1000;
      std::string file_name = boost::str (boost::format ("result_%06d.pcd") % tstamp);
      ROS_WARN_STREAM ("Writing "<<file_name<<"   framecounter="<<framecounter);
      
      std::string root_dir = ".";
      boost::filesystem::path root_path(root_dir);
      root_path /= "result_clouds";
      boost::filesystem::create_directory (root_path);
      pcl::io::savePCDFile ((root_path / file_name ).string (), state_cloud);
      
      std::string labeled_file_name = boost::str (boost::format ("labeled_%06d.pcd") % tstamp);
      boost::filesystem::path labeled_path(root_dir);
      labeled_path /= "labeled_clouds";
      boost::filesystem::create_directory (labeled_path);
      pcl::io::savePCDFile ((labeled_path / labeled_file_name ).string (), *combined_output, true);
      
      std::string ds_cloud_file_name = boost::str (boost::format ("ds_%06d.pcd") % tstamp);
      boost::filesystem::path ds_path(root_dir);
      ds_path /= "ds_clouds";
      boost::filesystem::create_directory (ds_path);
      pcl::io::savePCDFile ((ds_path / ds_cloud_file_name ).string (), *downsampled_cloud, true);
      
      std::string particle_file_name = boost::str (boost::format ("particles_%06d.pcd") % tstamp);
      boost::filesystem::path particle_path(root_dir);
      particle_path /= "particle_clouds";
      boost::filesystem::create_directory (particle_path);
      pcl::io::savePCDFile ((particle_path / particle_file_name ).string (), *particle_cloud, true);
    }
    
    t_viz += timer.getTime () - t_viz_start;
    ROS_INFO_STREAM_THROTTLE (txt_out_period,"===  Visualization time ="<<t_viz / frame_count <<" ms  ===");
    
    if (do_visualization)
      viewer->spinOnce ();
    r.sleep ();
    ros::spinOnce ();
    ++frame_count;
  }

  ROS_INFO("Quitting");
  return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////
void 
downsampleCloud(MsgCloudT::ConstPtr cloud_in, pcl::VoxelGrid< PointT >& voxel_grid, pcl::PointCloud< PointT >::Ptr& output)
{
  //Convert from ROS format
  InputCloudT::Ptr cloud_in_template (new InputCloudT);
  pcl::fromROSMsg (*cloud_in, *cloud_in_template);
  
  //Make a copy with the normal type
  CloudT::Ptr combined_cloud (new CloudT);
  //pcl::copyPointCloud (*cloud_in_template, *combined_cloud);
  combined_cloud = cloud_in_template;
  output.reset(new CloudT);
  
  voxel_grid.setDownsampleAllData (true);
  voxel_grid.setInputCloud (combined_cloud);
  voxel_grid.filter (*output);
}

//////////////////////////////////////////////////////////////////////////////////////////////
void 
cropCloud (CloudT::ConstPtr input_cloud, pcl::CropBox<PointT>& crop_box_filter,CloudT::Ptr &output)
{
  crop_box_filter.setInputCloud (input_cloud);
  output.reset(new CloudT);
  crop_box_filter.filter (*output);
}


//////////////////////////////////////////////////////////////////////////////////////////////
pcl::ModelCoefficients 
removeGroundPlane (CloudT::ConstPtr input_cloud, pcl::SACSegmentation<PointT>& seg, CloudT::Ptr &output)
{
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  seg.setInputCloud (input_cloud);
  seg.segment (*inliers, *coefficients);
  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud (input_cloud);
  extract.setIndices (inliers);
  extract.setNegative (true);
  
  CloudT::Ptr no_ground_cloud = boost::make_shared <CloudT> ();
    
  extract.filter (*no_ground_cloud);
  
  
  output.reset ( new CloudT );
  output->reserve (no_ground_cloud->size());
  
  float a = coefficients->values[0];
  float b = coefficients->values[1];
  float c = coefficients->values[2];
  float d = coefficients->values[3];
  
  if (d < 0.0f) {
    a = -a;b = -b;c = -c;d = -d;
  }
  
  for (CloudT::const_iterator it = no_ground_cloud->begin(); it != no_ground_cloud->end(); ++it)
    if (a*it->x + b*it->y + c*it->z + d > 0.0f)
      output->push_back(*it);
      
  return *coefficients;
}


//////////////////////////////////////////////////////////////////////////////////////////////
void 
initialSegmentation (const CloudT::ConstPtr combined_cloud, std::vector<pcl::PointIndices>& cluster_indices)
{
    // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<PointT>::Ptr tree = boost::make_shared<pcl::search::KdTree<PointT> >();
  tree->setInputCloud (combined_cloud);
  
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance (0.08f); 
  ec.setMinClusterSize (10);
  ec.setMaxClusterSize (400);
  ec.setSearchMethod (tree);
  ec.setInputCloud (combined_cloud);
  ec.extract (cluster_indices);
  
  // Initialize the objects to be tracked
  ROS_WARN_STREAM ("Number of detections = "<<cluster_indices.size ());
  ROS_WARN_STREAM ("==================================================");
  srand (static_cast<unsigned int> (time (0)));
  for(int j = 0; j < cluster_indices.size (); ++j) 
  {
    int ID = j+1;
    ID_to_label_[ID] = "Unknown";
    //Calc centroid
    float x=0, y=0, z=0;
    for (int i = 0; i < cluster_indices[j].indices.size (); ++i)
    {
      int idx = cluster_indices[j].indices[i];
      x += combined_cloud->points[idx].x;
      y += combined_cloud->points[idx].y;
      z += combined_cloud->points[idx].z;
    }
    boost::array<double,16> temp_pose { {1,0,0, x/cluster_indices[j].indices.size (),
                                         0,1,0, y/cluster_indices[j].indices.size (),
                                         0,0,1, z/cluster_indices[j].indices.size (),
                                         0,0,0,1} };    
    ID_to_pose_ [ID] = temp_pose;
    
    ROS_ERROR_STREAM ("Detected a "<< ID_to_label_[ID]<< ",  it has ID: "<< ID);
    //Generate a labeling color for this object
    
    uint32_t color;
    uint8_t r,g,b;
    r = static_cast<uint8_t>( (rand () % 256));
    g = static_cast<uint8_t> ( (rand () % 256));
    b = static_cast<uint8_t>( (rand () % 256));
    color = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
    ID_to_color_[ID] = color; 
  }
  ROS_WARN_STREAM ("==================================================");
  
}



//////////////////////////////////////////////////////////////////////////////////////////////
void 
initializeTracker (const CloudT::ConstPtr combined_cloud, 
                   pcl::tracking::MultiModelTracker<PointT, PosePointT>& tracker, 
                   const std::vector<pcl::PointIndices>& cluster_indices)
{
  ROS_WARN_STREAM ("Initializing tracker - tracking "<<ID_to_label_.size ()<< " objects");
  //Now iterate through detected objects, load their models and initialize tracker
  for (IDLabelMap::iterator label_itr = ID_to_label_.begin(); label_itr!=ID_to_label_.end(); ++label_itr)
  { // Pull out variables for clarity 
  std::string label = label_itr->second;
  int ID = label_itr->first;
  CloudT::Ptr cluster_points ( new CloudT );
  cluster_points->reserve (cluster_indices[ID-1].indices.size ());
  //Extract the points
  for (std::vector<int>::const_iterator pit = cluster_indices[ID-1].indices.begin (); pit != cluster_indices[ID-1].indices.end (); pit++) 
  { 
    cluster_points->push_back (combined_cloud->points[*pit]);
  }
  //Now initialize the tracker
  tracker.addCloudToTrack (cluster_points, ID);
  ROS_INFO_STREAM ("Done with "<<label);
  }
  

  tracker.setThreadCounts (number_of_threads_);
  tracker.setInputCloud (combined_cloud);
  bool tracked_frame = tracker.doTracking ();
  IDPoseMap::iterator pose_itr = ID_to_pose_.begin ();
  for ( ; pose_itr!=ID_to_pose_.end(); ++pose_itr)
  {
    int ID = pose_itr->first;
    boost::array<double,16> pose = pose_itr->second;
    Eigen::Affine3f pose_detector;
    int k = 0;
    for (int r = 0; r<4; ++r)
      for (int c = 0; c<4; ++c, ++k)
        pose_detector (r,c) = pose[k];
    //Pull out the initial tracked state
    pcl::tracking::ParticleXYZRPY initial_state = tracker.getResult (ID);
    ROS_WARN_STREAM ( "Initial pose for " << ID_to_label_.at(ID) << " = " << initial_state );
    //Calculate transform from tracked frame to obj detector frame
    Eigen::Affine3f pose_tracker = initial_state.toEigenMatrix ();
    Eigen::Affine3f tracked_to_detected = pose_tracker.inverse (Eigen::Affine) * pose_detector;
    ID_to_conversion_ [ ID ] = tracked_to_detected;
  }
  
}


//////////////////////////////////////////////////////////////////////////////////////////////
// Remove private parameters from the parameter server
void 
clearParams (ros::NodeHandle& n) 
{
   n.deleteParam ("source");
   n.deleteParam ("camera_topic");
}

//////////////////////////////////////////////////////////////////////////////////////////////
void 
keyboardCallback (const pcl::visualization::KeyboardEvent& event, void*)
{
  int key = event.getKeyCode ();
  if (event.keyUp ())    
    switch (key)
    {
      case (int)'1': show_no_ground_ = !show_no_ground_; break;
      case (int)'2': show_voxels_ = !show_voxels_; break;
      case (int)'3': show_result_ = !show_result_; break;
      case (int)'4': show_result_colored_ = !show_result_colored_; break;
      case (int)'5': show_tracks_ = !show_tracks_; break;
      case (int)'6': show_names_ = !show_names_; break;
      case (int)'7': show_particles_ = !show_particles_; break;
      case (int)'h': case (int)'H': show_help_ = !show_help_; break;
      default: break;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Callback for cloud data - label used is the source name
void 
pointcloudCallback (MsgCloudT::ConstPtr cloud_msg) 
{
  ROS_INFO_STREAM("Received message!");
  last_cloud_ = cloud_msg;
}

//////////////////////////////////////////////////////////////////////////////////////////////
void printText (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
  std::string on_str = "on";
  std::string off_str = "off";
  if (!viewer->updateText ("Press (1-n) to show different elements (h) to disable this", 5, 82, 12, 1.0, 1.0, 1.0,"hud_text"))
    viewer->addText ("Press (1-n) to show different elements (h) to disable this", 5, 82, 12, 1.0, 1.0, 1.0,"hud_text");
  
  std::string temp = "(1) Voxels with Ground Removed - currently " + ((show_no_ground_)?on_str:off_str);
  if (!viewer->updateText (temp, 5, 70, 10, 1.0, 1.0, 1.0, "no_ground_text"))
    viewer->addText (temp, 5, 70, 10, 1.0, 1.0, 1.0, "no_ground_text");
  
  temp = "(2) Voxels - currently "+ ((show_voxels_)?on_str:off_str);
  if (!viewer->updateText (temp, 5, 60, 10, 1.0, 1.0, 1.0, "voxel_text") )
    viewer->addText (temp, 5, 60, 10, 1.0, 1.0, 1.0, "voxel_text");
  
  temp = "(3) Tracked results - currently "+ ((show_result_)?on_str:off_str);
  if (!viewer->updateText (temp, 5, 50, 10, 1.0, 1.0, 1.0, "result_text") )
    viewer->addText (temp, 5, 50, 10, 1.0, 1.0, 1.0, "result_text");
  
  temp = "(4) Tracked result (colored) - currently "+ ((show_result_colored_)?on_str:off_str);
  if (!viewer->updateText (temp, 5, 40, 10, 1.0, 1.0, 1.0, "result_colored_text") )
    viewer->addText (temp, 5, 40, 10, 1.0, 1.0, 1.0, "result_colored_text");
  
  temp = "(5) Tracks - currently "+ ((show_tracks_)?on_str:off_str);
  if (!viewer->updateText (temp, 5, 30, 10, 1.0, 1.0, 1.0, "tracks_text") )
    viewer->addText (temp, 5, 30, 10, 1.0, 1.0, 1.0, "tracks_text");
  
  temp = "(6) Object names - currently "+ ((show_names_)?on_str:off_str);
  if (!viewer->updateText (temp, 5, 20, 10, 1.0, 1.0, 1.0, "names_text") )
    viewer->addText (temp, 5, 20, 10, 1.0, 1.0, 1.0, "names_text");
  
  temp = "(7) Particles - currently "+ ((show_particles_)?on_str:off_str);
  if (!viewer->updateText (temp, 5, 10, 10, 1.0, 1.0, 1.0, "particles_text") )
    viewer->addText (temp, 5, 10, 10, 1.0, 1.0, 1.0, "particles_text");
}

void removeText (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
  viewer->removeShape ("hud_text");
  viewer->removeShape ("voxel_text");
  viewer->removeShape ("no_ground_text");
  viewer->removeShape ("result_text");
  viewer->removeShape ("result_colored_text");
  viewer->removeShape ("tracks_text");
  viewer->removeShape ("names_text");
  viewer->removeShape ("particles_text");
}
