import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
from TF_network.DDPG_Network import DDPG
from TF_network.ReplayBuffer import ReplayBuffer
import tensorflow as tf
import copy
import time
import sys

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/ubuntu/catkin_ws/src/car/scripts'])


class SELF_Driving(object):
    count_no = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    action_deque = [np.array([0])]
    replaybuff = ReplayBuffer(20000)
    distance_list = []

    def __init__(self, is_training):
        self.a_dim = 1
        self.s_dim = 60
        self.is_training = is_training
        self.sess = tf.Session(config=self.config)
        self.ddpg = DDPG(self.sess, a_dim=self.a_dim, s_dim=self.s_dim, abound=np.array([[-0.85], [0.85]]))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.reward_list = []
        try:
            self.saver.restore(self.sess, 'model/model_test.ckpt')
        except:
            print('model error')
        self.step = 0
        self.current_state = None
        self.last_state = None
        self.action = [0]
        self.last_action = [0]

        self.move_cmd = Twist()
        rospy.init_node('listener', anonymous=False)
        rospy.Subscriber('/scan', LaserScan, self.callback)
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    def callback(self, data):
        self.count_no = self.count_no + 1
        if self.count_no == 8:
            self.count_no = 0
            self.one_step(data.ranges)

    def process_data(self, lidar_data):
        reset = False
        distance_data = 14.0 * np.array(lidar_data).ravel()[::-1]
        distance_data = np.where(distance_data > 20, 20, distance_data)
        step = distance_data.shape[0] // self.s_dim
        partition_index = [np.arange(i * step, (i + 1) * step) for i in range(self.s_dim - 1)]
        partition_index.append(np.arange((self.s_dim - 1) * step, distance_data.shape[0]))
        distance_data = np.array([np.min(distance_data[index]) for index in partition_index])

        reward = 8
        if np.min(lidar_data) < 0.16:
            reward = reward - 60
            reset = True

        focus_no = distance_data.shape[0] // 5
        sort_distance = np.sort(distance_data)
        focus_distance_mean = np.mean(sort_distance[:focus_no])
        reward = reward - 128.0 / np.power(2, focus_distance_mean)

        return distance_data, reward, focus_distance_mean, reset

    def one_step(self, lidar_data):
        self.step = self.step + 1
        self.current_state, reward, this_distance, reset = self.process_data(lidar_data)
        self.distance_list.append(this_distance)
        if reset:
            self.execute(np.array([0, 0]))
            if self.last_state is not None:
                self.reward_list.append(reward)
                self.replaybuff.add(copy.deepcopy(self.last_state),
                                    copy.deepcopy(self.last_action),
                                    copy.deepcopy(reward),
                                    copy.deepcopy(self.current_state))
            self.current_state = None
        else:
            if self.last_state is not None:
                self.reward_list.append(reward)
                self.replaybuff.add(copy.deepcopy(self.last_state),
                                    copy.deepcopy(self.last_action),
                                    copy.deepcopy(reward),
                                    copy.deepcopy(self.current_state))
            self.action = self.ddpg.choose_action(np.expand_dims(self.current_state, 0))
            car_action = self.clip_action(self.action)
            self.execute(car_action)
            self.self_car_train()
        self.last_state = copy.deepcopy(self.current_state)
        self.last_action = copy.deepcopy(self.action)

    def self_car_train(self):
        if (self.step + 1) % 600 == 0:
            self.execute(np.array([0, 0]))
            self.saver.save(self.sess, 'model/model_test.ckpt')

        if self.replaybuff.count() > 800:
            batch_samples = self.replaybuff.getBatch(4)
            self.ddpg.learn(batch_samples)

        if (self.step + 300) % 600 == 0:
            self.execute(np.array([0, 0]))
            self.saver.restore(self.sess, 'model/model_test.ckpt')
        # np.save('distance.npy', np.array(self.distance_list))

        if (self.step + 10) % 5000 == 0:
            self.execute(np.array([0, 0]))
            np.save('reward_list.npy', np.array(self.reward_list))

    def execute(self, control_cmd):
        self.move_cmd.linear.x = 1.4
        self.move_cmd.angular.z = control_cmd[0]
        if control_cmd.shape[0] > 1:
            self.move_cmd.linear.x = 0
        self.cmd_vel.publish(self.move_cmd)

    def clip_action(self, action):
        car_action = 4.8 * action
        return car_action


if __name__ == '__main__':
    try:
        self_driver = SELF_Driving(is_training=True)
        a = input()
    except:
        rospy.loginfo("Listener error")
