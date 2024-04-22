# Tenacious-Turtles


To establish an SSH connection between the robot and the host PC, follow these steps:
 
1. Open the terminal and run:
```
ssh ubuntu@IP_ADDRESS_OF_RASPI_ON_ROBOT
```
 
2. Update the `.bashrc` file on both devices:
 
On the robot:
```bash
echo "export ROS_MASTER_URI=http://IP_ADDRESS_OF_REMOTE_PC:11311" >> ~/.bashrc
echo "export ROS_HOSTNAME=IP_ADDRESS_OF_RASPI_ON_ROBOT" >> ~/.bashrc
```
 
On the host PC:
```bash
echo "export ROS_MASTER_URI=http://IP_ADDRESS_OF_REMOTE_PC:11311" >> ~/.bashrc
echo "export ROS_HOSTNAME=IP_ADDRESS_OF_REMOTE_PC" >> ~/.bashrc
```
 
3. Start the turtlebot bringup on the robot:
```
roslaunch turtlebot3_bringup turtlebot3_robot.launch
```
 
4. In another terminal on the host PC, run:
```
roscore
```
 
5. Start the robot by running the `master.launch` file:
```
roslaunch squirtle master.launch
```