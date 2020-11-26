import tensorflow as tf
import numpy as np
from DDPG_Network import DDPG
from ReplayBuffer import ReplayBuffer
from Fed_AGG import Federated_AGG
import setup_path 
import airsim
import time
import matplotlib.pyplot as plt
import copy
dict_list = []
car1_dict = {}
car1_dict['host_name'] = '192.168.8.202'
car1_dict['port'] = 22
car1_dict['username'] = 'ubuntu'
car1_dict['pwd'] = 'ubuntu'
car1_dict['target_model_path'] = '/home/ubuntu/catkin_ws/src/car/scripts/model/'
car1_dict['car_local_path'] = 'D:\\car1\\'
car2_dict = copy.deepcopy(car1_dict)
car2_dict['car_local_path'] = 'D:\\car2\\'
car2_dict['host_name'] = '192.168.8.202'

car3_dict = copy.deepcopy(car1_dict)
car3_dict['car_local_path'] = 'D:\\car3\\'
car3_dict['host_name'] = '192.168.8.202'

dict_list = [car1_dict,car2_dict,car3_dict]
#####################  hyper parameters  ####################

###############################  DDPG  ####################################
np.random.seed(19890214)

class SELF_CAR(object):
    def __init__(self,lidar_partition=60):
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()                     
        self.lidar_partition = lidar_partition # from left to right
        
        self.command_count = 0
        
        
        
    def execute_comand(self,control_array):
        self.car_controls.steering = control_array[0]
        self.car_controls.throttle = 0.35          
        self.car_controls.brake = 0
        self.client.setCarControls(self.car_controls)
        self.command_count = self.command_count + 1
        
    def reset_envs(self):
        self.command_count = 0
        self.client.reset()
        time.sleep(0.5)
        
    def get_state_reward(self,current_action):
        lidarData = self.client.getLidarData()
        if (len(lidarData.point_cloud) < 3):
            return 0,0,0
        distance_state = self._cords_to_distance(lidarData)
        collision_info = self.client.simGetCollisionInfo()
        car_state  = self.client.getCarState()
        reward, reset = self._get_award(distance_state,car_state,collision_info,current_action)
        return distance_state, reward, reset
    
    def _cords_to_distance(self,data):
        all_points = np.array(data.point_cloud, dtype=np.dtype('f4'))
        all_points = np.reshape(all_points, (int(all_points.shape[0]/3), 3))
        while all_points.shape[0] < self.lidar_partition:
            all_points = np.concatenate([all_points,all_points],axis=0)
        
        
        def np_dot(array1):
            return np.sum(array1*np.array([0,-1,0])) / np.sqrt(np.sum(np.square(array1)))
        def min_distance(aligned_array):
            if aligned_array.ndim < 2:
                aligned_array = np.expand_dims(aligned_array,0)
            return np.min( np.sqrt(np.sum(np.square(aligned_array),axis=1)) )

        all_cos_values = np.array(list( map( lambda x: np_dot(x), all_points)  ) )

        sequence_index = np.argsort(-all_cos_values)
        step = len(sequence_index) // self.lidar_partition ## this is a super_param
        aligned_index = [sequence_index[i*step:(i+1)*step] for i in range(0,self.lidar_partition-1)]
        aligned_index.append(sequence_index[(self.lidar_partition-1)*step:])
        distance_result = [ min_distance(all_points[i])  for i in  aligned_index ]
        return np.array(distance_result)


    def _get_award(self,distance_state,car_state, collision_info,current_action):
        reward = 8
        reset = False
        if collision_info.has_collided or (car_state.speed<0.5 and self.command_count>10):
            reward = reward - 60
            reset = True
        else:
            focus_no = distance_state.shape[0]//5
            sort_distance = np.sort(distance_state)
            fucus_dist_mean = np.mean(sort_distance[:focus_no])
            reward = reward - 128.0/np.power(2,fucus_dist_mean)
        return reward, reset

class Train(object):
    BUFFER_SIZE = 40000
    BATCH_SIZE = 32
 
         
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    def __init__(self, is_training=True):
        self.reward_list = []
        self.a_dim = 1
        self.s_dim = 60
        self.replaybuffer = ReplayBuffer(self.BUFFER_SIZE)
        self.is_training = is_training   
        self.self_car = SELF_CAR(self.s_dim)
        self.sess = tf.Session(config=self.config)
        self.ddpg = DDPG(self.sess, a_dim=self.a_dim, s_dim=self.s_dim, abound=np.array([[-0.85],[0.85]]))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.fed_agg = Federated_AGG(self.sess,dict_list)
        self.fed_agg.upload()
        self.federated_count = 1
        try:
            self.saver.restore(self.sess,'model/model_test.ckpt')           
        except:
            print('cannot load model')
            

            
    def execute_fed_process(self):
        self.federated_count = self.federated_count + 1
        self.fed_agg.download()
        self.fed_agg.aggregate()
        self.fed_agg.upload()
        print('Model update success!!!, update index----' + str(self.federated_count)+'----')     
        
    def online_train(self): 
        var = 0.00
        epsilon_decay = 1/30000.0
        step = 0   
        while True:
            reset = True
            for round_index in range(5000):
                if reset:
                    self.self_car.reset_envs()
                    
                    current_state, reward, reset = self.self_car.get_state_reward([0])                    
                    if type(current_state) == int:
                        break
                    reset = False
                    total_reward = 0
                      
                    
                var = var - epsilon_decay
                step += 1   
                
                action = self.ddpg.choose_action(np.expand_dims(current_state,0))[0]
                
                if self.is_training and var > 0:
                    action = np.clip(np.random.normal(action, var),-0.85,0.85) # steering, throttle, brake
                                
                self.self_car.execute_comand([float(action)])
                
                start = time.time()
                
                if (step + 1) % 200 == 0:
                    self.saver.save(self.sess,'model/model_test.ckpt') 
                    self.execute_fed_process()
                    self.saver.restore(self.sess,'model/model_test.ckpt')
                    np.save('simu_reward.npy',np.array(self.reward_list))
                    reset = True
                    break
                else:
                    #if self.replaybuffer.count() > 5000:
                        #batch_samples = self.replaybuffer.getBatch(self.BATCH_SIZE)
                        #self.ddpg.learn(batch_samples)                  
                    
                    remaining_time = 0.1 - (time.time() - start)
                    if remaining_time > 0:
                        time.sleep(remaining_time)                                
                    next_state, reward, reset = self.self_car.get_state_reward(action)
                    self.reward_list.append(reward)
                    if type(next_state) == int:
                        print(step)
                        break                
                    self.replaybuffer.add( copy.deepcopy(current_state),copy.deepcopy(action),copy.deepcopy(reward),copy.deepcopy(next_state)  )
                        
                    total_reward +=   reward
                    current_state = next_state                 
                if reset:
                    pass
                    #print('this round with reward', total_reward, ' and var: ',var)
                   
a = Train(is_training=True)
a.online_train()
