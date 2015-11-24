#Message for publishing tracked models, which is used for keyframe detection
Header header
string[] object_strings
geometry_msgs/Pose[] object_poses
uint32[] labels
float64[] table_coefficients
sensor_msgs/PointCloud2 labeled_model_cloud
sensor_msgs/PointCloud2 raw_cloud
sensor_msgs/PointCloud2 tracks
sensor_msgs/PointCloud2 states
geometry_msgs/Pose[] object_poses_init

#Old pose type for compatibility
float64[] poses