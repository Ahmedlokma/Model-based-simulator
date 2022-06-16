import logging
import time
import configparser
class _Config:
    def __init__(self):
        self._init_logging_handler()
        self.cuda_device = 4
        self.eos_m_token = 'EOS_M'       
        self.beam_len_bonus = 0.6

        self.mode = 'unknown'
        self.m = 'TSD'
        self.prev_z_method = 'none'
        self.dataset = 'unknown'

        self.seed = 0
  
    def init_handler(self, m):
        init_method = {

 
            'tsdf-usr': self._usr_tsdf_init
        }
        init_method[m]()

 

    def _usr_tsdf_init(self):
        self.vocab_size = 8000
        self.embedding_size = 50
        self.hidden_size = 50
        self.lr = 0.003
        self.lr_decay = 0.5
        self.layer_num = 1
        self.z_length = 16
        self.max_ts = 50
        self.early_stop_count = 5
        self.cuda = True
        self.degree_size = 1



        self.split = (9, 1, 0)
        # self.root_dir = "/Users/ahmedlokma/Desktop/user-simulator-master/sequicity_user"
 

        # self.split = (9, 1, 1)
        # self.root_dir = "/Users/ahmedlokma/Desktop/user-simulator-master/sequicity_user"
        # self.model_path = self.root_dir + '/models/trainmodel.pkl'
        # self.result_path = self.root_dir + '/results/trainmodel.csv'
        # self.vocab_path = self.root_dir + '/vocab/vocab-trainmodel.pkl'

        # self.data = '/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/example2.json'
        # self.entity = '/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/example.json'
        # self.db = '/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/trainmodel.json'
        
        self.root_dir = "/Users/ahmedlokma/Desktop/user-simulator-master/sequicity_user"
        self.model_path = self.root_dir + '/models/wed3000000.pkl'
        self.result_path = self.root_dir + '/results/wed3000000.csv'
        self.vocab_path = self.root_dir + '/vocab/wed3000000.pkl'

        self.data = '/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/Guc_Dataset_Dialogue_act22.json'
        self.entity = '/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/Guc_Dataset_Entity_Sorted22.json'
        self.db = '/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/Guc_Dataset22.json'


        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate'
        self.glove_path = '../sequicity/data/glove/glove.6B.50d.txt'
        self.batch_size = 32
        self.dropout_rate = 0.5
        self.epoch_num = 100 # triggered by early stop
        self.rl_epoch_num = 1
        self.spv_proportion = 100
        self.new_vocab = True
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.use_positional_embedding = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False
    
    
 

    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s

    def _init_logging_handler(self):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        stderr_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('./log/log_{}.txt'.format(current_time))
        # file_handler = logging.FileHandler('sequicity_user/log/log_{}.txt'.format(current_time))
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()

