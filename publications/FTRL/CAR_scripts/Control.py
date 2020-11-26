import rospy
from geometry_msgs.msg import Twist  


class Control(object):
    def __init__(self):
        rospy.init_node('car_move', anonymous=False)

        rospy.loginfo("To stop car CTRL + C") 

        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)


    def shutdown(self):
        rospy.loginfo("Stop car")
        self.cmd_vel.publish(Twist())  
        rospy.sleep(1) 


    def publish_cmd(self,cmd):
        self.cmd_vel.publish(cmd)



