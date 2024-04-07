#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

class GoForward:
    
    def forward(self, lin, ang):
        
        rospy.loginfo("Moving - lin : {} ang : {}".format(lin,ang))
        
        move_cmd = Twist()
        move_cmd.linear.x = lin
        move_cmd.angular.z = ang
        self.cmd_vel.publish(move_cmd)

    def __init__(self):
        
        rospy.init_node('GoForward', anonymous=False)
        rospy.loginfo("To stop TurtleBot CTRL + C")
        rospy.on_shutdown(self.shutdown)
        
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        r = rospy.Rate(10)
	
        while not rospy.is_shutdown():
            t0 = rospy.get_rostime().secs
            while(t0 + 5 >= rospy.get_rostime().secs):
                self.forward(0.3,0)
                
            t0 = rospy.get_rostime().secs
            while(t0 + 1 >= rospy.get_rostime().secs):
                self.forward(0,1)

            t0 = rospy.get_rostime().secs
            while(t0 + 5 >= rospy.get_rostime().secs):
                self.forward(0.2,0)
            break
                        
        
    def shutdown(self):
        rospy.loginfo("Stop TurtleBot")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)
 
if __name__ == '__main__':

    try:
        move = GoForward()
    except:
        rospy.loginfo("Node terminated.")