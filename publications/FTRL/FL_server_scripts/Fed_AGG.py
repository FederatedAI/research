import paramiko
import tensorflow as tf
from DDPG_Network import DDPG
import numpy as np

class Federated_AGG(object): 
    def __init__(self, sess,dict_list):
        self.dict_list = dict_list
        self.federated_model_path = 'D:\\DDPG_final\\model\\'
        self.simulator_model_path = 'D:\\DDPG_final\\model\\'
        self.model_file_list = ['checkpoint',
                                    'model_test.ckpt.data-00000-of-00001',
                                    'model_test.ckpt.index',
                                    'model_test.ckpt.meta']
        self.sess = sess
        self.all_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()        
        
    def connect(self,sg_car_dict):
        transport = paramiko.Transport((sg_car_dict['host_name'],sg_car_dict['port']))
        transport.connect(username=sg_car_dict['username'],password=sg_car_dict['pwd'])
        return transport
 
    def close(self,transport):
        transport.close()
        
    def upload(self): # diliver files to all cars
        for sg_car_dict in self.dict_list:
            transport = self.connect(sg_car_dict)
            sftp = paramiko.SFTPClient.from_transport(transport)
            for file_name in self.model_file_list:
                local_file_path = self.federated_model_path + str(file_name)
                target_file_path = sg_car_dict['target_model_path'] + str(file_name)
                sftp.put(local_file_path, target_file_path, confirm=True)
                sftp.chmod(target_file_path, 0o755)
                
        
    def download(self): # get all files from all cars
        for sg_car_dict in self.dict_list:
            transport = self.connect(sg_car_dict)
            sftp = paramiko.SFTPClient.from_transport(transport) 
            for file_name in self.model_file_list:
                local_file_path = sg_car_dict['car_local_path']  + str(file_name) 
                target_file_path = sg_car_dict['target_model_path'] + str(file_name)                
                sftp.get(target_file_path, local_file_path)

    def aggregate(self):
        agg_model_file_list = []
        for sg_car_dict in self.dict_list:
            agg_model_file_list.append(sg_car_dict['car_local_path']+'model_test.ckpt')
        agg_model_file_list.append(self.simulator_model_path+'model_test.ckpt')
        
        model_vars = []
        print('-------------------------------------')
        for sg_car,sg_model_path in  zip(['car1','car2','car3','simulator'],agg_model_file_list):
            print('------loading '+ str(sg_car) + ' model succeeded')
            self.saver.restore(self.sess,sg_model_path)
            model_vars.append(self.sess.run(self.all_vars)  )  
            
        all_sign = []
        for model_var, var1, var2, var3,var4 in zip(self.all_vars,model_vars[0],model_vars[1],model_vars[2],model_vars[3]):
            all_sign.append(tf.assign(model_var,(var1+var2+var3+var4)/4.0))
        
        self.sess.run(all_sign)
        federated_model_path = self.federated_model_path + 'model_test.ckpt'
        self.saver.save(self.sess,federated_model_path)
        print('------Federated Model Generated')
        print('-------------------------------------')
    def __del__(self):
        self.close()

