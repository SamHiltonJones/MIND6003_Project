# SnagSpotter
## Objective
SnagSpotter is an autonomous turtlebot robot that can traverse an environment to compare a reference point cloud to a real point cloud generated from a depth camera. Upon observing differences between the point clouds, they are updated to the map. Therefore, we recieve an updated reference point cloud with obstacles added. This can be seen through a flowchart of our processes:

![alt text](https://github.com/SamHiltonJones/MIND6003_Project/blob/main/images/flowchart.png?raw=true)
## Running the code
This code runs within a Gazebo simulation running ROS2. This dependancy will have to be install first. By running the following commands the simulation with SnagSpotter will run:
```bash
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
ros2 launch office_robot_pkg gazebo_world.launch.py
```
This will launch the 'Control Panel', Rviz2 and Gazebo with our sample office environment. The Rviz2 window has the pre-generated cost map from the slam_toolbox in which we use nav2_bringup to allow autonomous navigation. 
## The Navigation
To navigate the robot use the Rviz2 window and select the Pose Estimation button, select and point the arrow in the direction the robot is facing in the environment. Next use the Goal Pose button and select and point the arrow in the direction of the goal pose. This generates a path that the robot will follow based on the cost map.
## The Control Panel
The Control Panel starts with three buttons, 'Detect Additions to Environment', 'Detect Removals from Environment' and 'Detect all changes To Environment'. These are the three modes of SnagSpotter, you must select one before any point cloud alignment takes place. Next is the message box, this is used to log changes for a later button. The 'Current Comparison' button allows visualisation of the reference point cloud vs the depth camera point cloud. 'View Differences' shows which significant clusters has been detected with DBScan between the point clouds. 'View Updated Map' shows the reference map with the significant clusters added, 'Generate Heat Map' generates a single frame of the heatmap of chanegs to the reference point cloud. 'Start/Stop Generating Heat Map' creates a heatmap frame every second of the changes and upon pressing again stops this generation creating a video.avi of these frames to give a dynamic view of the changes to the environment. 'Log Changes' uses K-Means clustering to cluster the changes into difference objects and logs this to the message box.
## send_server folder
This is the folder that would be sent to a server in a real-world scenario. This contains the updated map, the heatmap video, a list of objects detected and viewer position using the fine-tuned YOLOv8 model, the images used to classify the objects with a bounding box from this model and the clustered changes with a figure of these changes. This could be useful to a facility manager for example  to detect where changes are occuring within the environment or a construction team to identify indoor snagging issues.
## How this works
The robot is placed in a sample 'office' environment that has 3 desks and 3 chairs added and 1 box removed opposed to the reference point cloud:

![alt text](https://github.com/SamHiltonJones/MIND6003_Project/blob/main/images/gazebo_simulation.png?raw=true)

The robot takes a reference point cloud and must estimate where it is within this point cloud. By using matrix geometry a virtual camera is placed within this reference point cloud to have a sample of what the robot might be observing. 

![alt text](https://github.com/SamHiltonJones/MIND6003_Project/blob/main/images/virtual_camera.png?raw=true)

Next, it compares this to the actual depth camera point cloud, which will be misaligned and hence an ICP method is used to align them. This takes the following misalignment:

![alt text](https://github.com/SamHiltonJones/MIND6003_Project/blob/main/images/misaligned_example.png?raw=true)

and produces the following aligned version:

![alt text](https://github.com/SamHiltonJones/MIND6003_Project/blob/main/images/aligned_example.png?raw=true)

This allows us to use a DBScan method to detect significant clusters that are different, this gives the following updated map:

![alt text](https://github.com/SamHiltonJones/MIND6003_Project/blob/main/images/updated_map.png?raw=true)

where we have added the 6 objects and removed the box that is missing. 
In another example where we have 4 obejcts added and 1 removed we use K-Means clustering to cluster these objects and log the changes:

![alt text](https://github.com/SamHiltonJones/MIND6003_Project/blob/main/images/clustered_changes.png?raw=true)

Finally, we use a YOLOv8 model that we fine-tune on images of tables, chairs and sofas and use this to create a bounding box around each object, reporting this to the server:

![alt text](https://github.com/SamHiltonJones/MIND6003_Project/blob/main/send_server/detected_objects/detected_objects_1.jpg?raw=true)
