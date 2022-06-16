
from fcntl import lockf
import sys
from unittest import result
import torch
import random
import numpy as np
import nltk 
import json
import difflib
from numpy import vectorize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
nltk.download('wordnet')
nltk.download('omw-1.4')

sys.path.append('/Users/ahmedlokma/Desktop/user-simulator-master/')
from torch.optim import Adam, RMSprop
import string
from torch.autograd import Variable
from sequicity_user.reader import CamRest676Reader, get_glove_matrix
from sequicity_user.config import global_config as cfg
from sequicity_user.reader import  User_Simulator_Reader
from sequicity_user.tsd_net import TSD,  nan
from sequicity_user.reader import pad_sequences
import argparse, time
from sequicity_user.metric import CamRestEvaluator, KvretEvaluator
import logging
import pdb
from nltk.tokenize import word_tokenize
from random import randint
from difflib import SequenceMatcher
from nltk.tokenize.treebank import TreebankWordDetokenizer

class Model:
   
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, dataset):

        self.olide_counter = 0
        self.oldie_index = []
        self.oldie = []
        reader_dict = {
            'camrest': CamRest676Reader,
            'usr': User_Simulator_Reader
        }
        model_dict = {
            'TSD':TSD
        }
        evaluator_dict = {
            'usr': CamRestEvaluator

        }
        self.dataset = dataset
        self.reader = reader_dict[dataset]()
        self.m = model_dict[cfg.m](embed_size=cfg.embedding_size,
                               hidden_size=cfg.hidden_size,
                               vocab_size=cfg.vocab_size,
                               layer_num=cfg.layer_num,
                               dropout_rate=cfg.dropout_rate,
                               z_length=cfg.z_length,
                               max_ts=cfg.max_ts,
                               beam_search=cfg.beam_search,
                               beam_size=cfg.beam_size,
                               eos_token_idx=self.reader.vocab.encode('EOS_M'),
                               vocab=self.reader.vocab,
                               teacher_force=cfg.teacher_force,
                               degree_size=cfg.degree_size,
                               reader=self.reader)
        self.EV = evaluator_dict[dataset] # evaluator class
        # if cfg.cuda: self.m = self.m.cuda()
        self.optim = Adam(lr=cfg.lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),weight_decay=5e-5)
        self.base_epoch = -1

    def _convert_batch(self, py_batch, prev_z_py=None):
        # py_batch = py_batch
        u_input_py = py_batch['user']
        u_len_py = py_batch['u_len']
        kw_ret = {}
        if cfg.prev_z_method == 'concat' and prev_z_py is not None:
            for i in range(len(u_input_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    u_input_py[i] = prev_z_py[i][:idx + 1] + u_input_py[i]
                else:
                    u_input_py[i] = prev_z_py[i] + u_input_py[i]
                u_len_py[i] = len(u_input_py[i])
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size:
                        prev_z_py[i][j] = 2 #unk
        elif cfg.prev_z_method == 'separate' and prev_z_py is not None:
            # print(py_batch['user'])
            for i in range(len(prev_z_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    prev_z_py[i] = prev_z_py[i][:idx + 1]
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size:
                        prev_z_py[i][j] = 2 #unk
            # print(py_batch['user'])
            prev_z_input_np = pad_sequences(prev_z_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
            prev_z_len = np.array([len(_) for _ in prev_z_py])
            prev_z_input = (Variable(torch.from_numpy(prev_z_input_np).long()))
            kw_ret['prev_z_len'] = prev_z_len
            kw_ret['prev_z_input'] = prev_z_input
            kw_ret['prev_z_input_np'] = prev_z_input_np

        degree_input_np = np.array(py_batch['degree'])
        # pdb.set_trace()
        u_input_np = pad_sequences(u_input_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
        z_input_np = pad_sequences(py_batch['bspan'], padding='post').transpose((1, 0))
        m_input_np = pad_sequences(py_batch['response'], cfg.max_ts, padding='post', truncating='post').transpose(
            (1, 0))


        u_len = np.array(u_len_py)
        m_len = np.array(py_batch['m_len'])

        degree_input = (Variable(torch.from_numpy(degree_input_np).float()))
        u_input = (Variable(torch.from_numpy(u_input_np).long()))
        z_input = (Variable(torch.from_numpy(z_input_np).long()))
        m_input = (Variable(torch.from_numpy(m_input_np).long()))

        # kw_ret['z_input_np'] = z_input_np

        if 'goal' in py_batch:
            g_input_np = pad_sequences(py_batch['goal'], cfg.z_length, padding='post').transpose((1, 0))
            g_input_len = np.array([len(_) for _ in py_batch['goal']])
            g_input = (Variable(torch.from_numpy(g_input_np).long()))
            kw_ret['g_input_np'] = g_input_np
            kw_ret['g_input_len'] = g_input_len
            kw_ret['g_input'] = g_input

        return u_input, u_input_np, z_input, m_input, m_input_np,u_len, m_len,  \
               degree_input, kw_ret

    def train(self):
        lr = cfg.lr
        prev_min_loss, early_stop_count = 1 << 30, cfg.early_stop_count
        train_time = 0
        for epoch in range(cfg.epoch_num):
            sw = time.time()
            if epoch <= self.base_epoch:
                continue
            self.training_adjust(epoch)
            self.m.self_adjust(epoch)
            sup_loss = 0
            sup_cnt = 0
            data_iterator = self.reader.mini_batch_iterator('train')
            optim = self.optim
            for iter_num, dial_batch in enumerate(data_iterator):
                turn_states = {}
                prev_z = None
                for turn_num, turn_batch in enumerate(dial_batch):
                     # print('Sys: ' + self.reader.vocab.sentence_decode(m_idx[0], eos='EOS_M'))
                    # lok = self.NLG(turn_batch) 
                    if cfg.truncated:
                        logging.debug('iter %d turn %d' % (iter_num, turn_num))
                    optim.zero_grad()
                    u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                    m_len, degree_input, kw_ret \
                        = self._convert_batch(turn_batch, prev_z)
                    # pdb.set_trace()

                    loss, pr_loss, m_loss, turn_states = self.m(u_input=u_input, z_input=z_input,
                                                                        m_input=m_input,
                                                                        degree_input=degree_input,
                                                                        u_input_np=u_input_np,
                                                                        m_input_np=m_input_np,
                                                                        turn_states=turn_states,
                                                                        u_len=u_len, m_len=m_len, 
                                                                        mode='train', 
                                                                        model=self.dataset, **kw_ret)
                    loss.backward(retain_graph=turn_num != len(dial_batch) - 1)
                    grad = torch.nn.utils.clip_grad_norm_(self.m.parameters(), 5.0)
                    optim.step()
                    # pdb.set_trace()
                    sup_loss += loss.data.cpu().numpy()
                    sup_cnt += 1

                    prev_z = turn_batch['bspan']

            logging.info(
                'loss:{} pr_loss:{} m_loss:{} grad:{}'.format(loss.data,
                                                               pr_loss.data,
                                                               m_loss.data,
                                                               grad))

            epoch_sup_loss = sup_loss / (sup_cnt + 1e-8)
            train_time += time.time() - sw
            logging.info('Traning time: {}'.format(train_time))
            logging.info('avg training loss in epoch %d sup:%f' % (epoch, epoch_sup_loss))

            valid_sup_loss, valid_unsup_loss = self.validate()
            logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            logging.info('time for epoch %d: %f' % (epoch, time.time()-sw))
            valid_loss = valid_sup_loss + valid_unsup_loss
            # logging.info('saving model...')
            # self.save_model(epoch)
            if valid_loss <= prev_min_loss:
                logging.info('saving model...')
                self.save_model(epoch)
                prev_min_loss = valid_loss
                early_stop_count = cfg.early_stop_count
            else:
                early_stop_count -= 1
                lr *= cfg.lr_decay
                if not early_stop_count:
                    break
                self.optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),
                                  weight_decay=5e-5)
                logging.info('early stop countdown %d, learning rate %f' % (early_stop_count, lr))

    def eval(self, data='test'):
        print("EVAAAAAAAAAAL LINE 246")
        self.m.eval()
        self.reader.result_file = None
        data_iterator = self.reader.mini_batch_iterator(data)
        mode = 'test' if not cfg.pretrain else 'pretrain_test'
        for batch_num, dial_batch in enumerate(data_iterator):
            turn_states = {}
            prev_z = None
            for turn_num, turn_batch in enumerate(dial_batch):
                print("___________________________________")
                print(turn_num)
                print("___________________________________")
                u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                m_len, degree_input, kw_ret \
                    = self._convert_batch(turn_batch, prev_z)
                lok = self.NLG(turn_batch)    
                # pdb.set_trace()
                m_idx, z_idx, turn_states = self.m(mode=mode, u_input=u_input, u_len=u_len, z_input=z_input,
                                                   m_input=m_input,
                                                   degree_input=degree_input, u_input_np=u_input_np,
                                                   m_input_np=m_input_np, m_len=m_len, turn_states=turn_states,
                                                   dial_id=turn_batch['dial_id'], **kw_ret)
                # print('Sys: ' + self.reader.vocab.sentence_decode(m_idx[0], eos='EOS_M'))
                # print('Slots: ' + self.reader.vocab.sentence_decode(z_idx[0], eos='EOS_Z2'))
                # filled_sent = self.fill_sentence(m_idx, z_idx)
                # print('Sys: ' + filled_sent)
                # print('-------------------------------------------------------\n')

                # pdb.set_trace()

                self.reader.wrap_result(turn_batch,lok, z_idx, prev_z=prev_z)
                prev_z = z_idx
        ev = self.EV(result_path=cfg.result_path)
        print(cfg.result_path)
        # time.sleep(50)
        res = ev.run_metrics()
        self.m.train()
        return res
    
    def NLG(self , batch): 
     z_results = []
     generated_results = []
     z = ""
    #  dialogue_act = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/example2.json','rb')
     dialogue_act = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/Guc_Dataset_Dialogue_act22.json','rb')
     dataList= json.load(dialogue_act)
     results_counter = 0
     print(len(batch['user']))
    #  time.sleep(30)
     for i in range(len(batch['user'])): 

      responses = []  
      responses2 = []
      index2  = 0 
      original_response = self.reader.vocab.sentence_decode(batch['response'][i], eos='EOS_M')
      print(original_response)
    #   time.sleep(40)
      slots = self.reader.vocab.sentence_decode(batch['bspan'][i])
      print(slots)
      
    #   print(original_response)
      token = '<unk>' 
      slots = slots.replace(token , "")
      slots = slots.split(',')

      found = False # used in multiple 
      mutliple = False 
      total = len(slots)
      found2 = False # used in multiple
      l =0
      while l < total :
       if(slots[l] ==" "):
           del slots[l]
       l = l +1    
      counter = len(slots)
      print(counter)  
    #   print(slots)
      
      if(counter > 1):
        mutliple = True
        # print(True) 
      if(mutliple == True):
        for index in dataList : 
         found == False  
         found2 == False   
         counter = len(slots)
         for item in index['dial']:
          for z in  item['A']['slu'] :
              s = z['slots'][0][0]
              if(found == True):
                 j = j+1
              for j in range(len(slots)):
               x = slots[j].strip()
               if(s == x):
                counter = counter - 1
                found = True
                
               
               if(found == True and s != x) :
                  
                  found = False 
                  found2 = True 
                  # break 

               if(found == True and counter > 0 and found2 == False ):
                  break 
               if (found == True and counter == 0):
                  temp_res = item['A']['transcript'] 
                  # print(temp_res)

                  generated_results.insert(results_counter,temp_res)
                  z_results.insert(results_counter,original_response)
                  results_counter = results_counter + 1
              
      else :
           for j in range(len(slots)): 
            x = slots[j].strip()
            for index in dataList :
             counter = len(slots)     
             for item in index['dial']:
              for z in  item['A']['slu'] :
               s = z['slots'][0][0].lower()
               print(s)
               print(x)
               print(original_response)
               print(item['A']['transcript'])
            #    time.sleep(10)
      
               if(s == x and mutliple == False):
                temp_res = item['A']['transcript']
                responses.insert(index2 , temp_res)
                responses2.insert(index2,s)
                index2 = index2 + 1 
                
           
      if(len(responses) > 1):  
          value = randint(0, len(responses)-1)
      elif (len(responses) == 0) :
         
          value = -1
      else :
         value = 0
      if value != -1:
          ccc = -1 
          sentences2 = []
          sentences2.insert(0,original_response) 
          sentences2.insert(1,responses[value])  
          cleaned = list(map(self.clean_string,sentences2))
          print(cleaned)
          vectorizer = CountVectorizer().fit_transform(cleaned)
          vectors = vectorizer.toarray()
          csim = cosine_similarity(vectors)
          print(csim)
          cc = self.cosine_sim_vectors(vectors[0],vectors[1])
        #   cc = SequenceMatcher(None, original_response,responses[value]).ratio()
          for c in range(len(responses)):
              sentences2 = []
              sentences2.insert(0,original_response) 
              sentences2.insert(1,responses[c])  
              cleaned = list(map(self.clean_string,sentences2))
              print(cleaned)
              vectorizer = CountVectorizer().fit_transform(cleaned)
              vectors = vectorizer.toarray()
              csim = cosine_similarity(vectors)
              print(csim)
              v = self.cosine_sim_vectors(vectors[0],vectors[1])
            #   v = SequenceMatcher(None, original_response,responses[c]).ratio()
              if(v >= cc and x == responses2[c]):
                  cc = v
                  ccc = c
            #   elif( v == c):
            #       cc = v
            #       ccc = c

          z_results.insert(results_counter,original_response)        
          generated_results.insert(results_counter,responses[ccc]) 
          results_counter += 1 
 
 
  

        
     return generated_results 
    def clean_string (self,text) :
     text = ''.join( [ word for word in text if word not in string.punctuation ])
     text = text.lower()
     text = ' '.join([word for word in text.split() if word not in stopwords])
     return text 
    def cosine_sim_vectors (self,vec1 , vec2):
     print("helllo")
     vec1 = vec1.reshape(1,-1)
     vec2 = vec2.reshape(1,-1)
     return cosine_similarity(vec1 , vec2 )[0][0]

    def validate(self, data='dev'):
        self.m.eval()
        data_iterator = self.reader.mini_batch_iterator(data)
        sup_loss, unsup_loss = 0, 0
        sup_cnt, unsup_cnt = 0, 0
        for dial_batch in data_iterator:
            turn_states = {}
            for turn_num, turn_batch in enumerate(dial_batch):
                u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                m_len, degree_input, kw_ret \
                    = self._convert_batch(turn_batch)

                loss, pr_loss, m_loss, turn_states = self.m(u_input=u_input, z_input=z_input,
                                                                    m_input=m_input,
                                                                    turn_states=turn_states,
                                                                    degree_input=degree_input,
                                                                    u_input_np=u_input_np, m_input_np=m_input_np,
                                                                    u_len=u_len, m_len=m_len, mode='train',**kw_ret)
                sup_loss += loss.data
                sup_cnt += 1
                logging.debug(
                    'loss:{} pr_loss:{} m_loss:{}'.format(loss.data, pr_loss.data, m_loss.data))

        sup_loss /= (sup_cnt + 1e-8)
        unsup_loss /= (unsup_cnt + 1e-8)
        self.m.train()
        print('result preview...')

        if cfg.dataset == 'usr_act':
            self.eval_act_classfier()
        else:
            self.eval()
        return sup_loss, unsup_loss

    def reinforce_tune(self):
        lr = cfg.lr
        self.optim = Adam(lr=cfg.lr, params=filter(lambda x: x.requires_grad, self.m.parameters()))
        prev_min_loss, early_stop_count = 1 << 30, cfg.early_stop_count
        for epoch in range(self.base_epoch + cfg.rl_epoch_num + 1):
            mode = 'rl'
            if epoch <= self.base_epoch:
                continue
            epoch_loss, cnt = 0,0
            data_iterator = self.reader.mini_batch_iterator('train')
            optim = self.optim #Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()), weight_decay=0)
            for iter_num, dial_batch in enumerate(data_iterator):
                turn_states = {}
                prev_z = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    optim.zero_grad()
                    u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                    m_len, degree_input, kw_ret \
                        = self._convert_batch(turn_batch, prev_z)
                    loss_rl = self.m(u_input=u_input, z_input=z_input,
                                m_input=m_input,
                                degree_input=degree_input,
                                u_input_np=u_input_np,
                                m_input_np=m_input_np,
                                turn_states=turn_states,
                                dial_id=turn_batch['dial_id'],
                                u_len=u_len, m_len=m_len, mode=mode, **kw_ret)

                    if loss_rl is not None:
                        loss = loss_rl #+ loss_mle * 0.1
                        loss.backward()
                        grad = torch.nn.utils.clip_grad_norm_(self.m.parameters(), 2.0)
                        optim.step()
                        epoch_loss += loss.data.cpu().numpy()[0]
                        cnt += 1
                        logging.debug('{} loss {}, grad:{}'.format(mode,loss.data[0],grad))

                    prev_z = turn_batch['bspan']

            epoch_sup_loss = epoch_loss / (cnt + 1e-8)
            logging.info('avg training loss in epoch %d sup:%f' % (epoch, epoch_sup_loss))

            valid_sup_loss, valid_unsup_loss = self.validate()
            logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            valid_loss = valid_sup_loss + valid_unsup_loss

            #self.save_model(epoch)

            if valid_loss <= prev_min_loss:
                self.save_model(epoch)
                prev_min_loss = valid_loss
            else:
                early_stop_count -= 1
                lr *= cfg.lr_decay
                if not early_stop_count:
                    break
                logging.info('early stop countdown %d, learning rate %f' % (early_stop_count, lr))

    def save_model(self, epoch, path=None, critical=False):
        if not path:
            path = cfg.model_path
        if critical:
            path += '.final'
        all_state = {'lstd': self.m.state_dict(),
                     'config': cfg.__dict__,
                     'epoch': epoch}
        torch.save(all_state, path)

    def load_model(self, path=None):
        if not path:
            path = cfg.model_path
        all_state = torch.load(path, map_location='cpu')
        # pdb.set_trace()
        self.m.load_state_dict(all_state['lstd'])
        self.base_epoch = all_state.get('epoch', 0)

    def training_adjust(self, epoch):
        return

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def load_glove_embedding(self, freeze=False):
        initial_arr = self.m.u_encoder.embedding.weight.data.cpu().numpy()
        embedding_arr = torch.from_numpy(get_glove_matrix(self.reader.vocab, initial_arr))

        self.m.u_encoder.embedding.weight.data.copy_(embedding_arr)
        self.m.z_decoder.emb.weight.data.copy_(embedding_arr)
        self.m.m_decoder.emb.weight.data.copy_(embedding_arr)

    def count_params(self):

        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = sum([np.prod(p.size()) for p in module_parameters])

        print('total trainable params: %d' % param_cnt)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-model')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.init_handler(args.model)
    cfg.dataset = args.model.split('-')[-1]

    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            else:
                v = dtype(v)
            setattr(cfg, k, v)

    logging.info(str(cfg))
   
    cfg.mode = args.mode

    torch.manual_seed(cfg.seed)
    
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    m = Model(args.model.split('-')[-1])
    m.count_params()
    if args.mode == 'train':
        m.load_glove_embedding()
        m.train()
    elif args.mode == 'adjust':
        m.load_model()
        m.train()
    elif args.mode == 'test':
        m.load_model()
        if cfg.dataset == 'usr_act':
            m.eval_act_classfier()
        else:
            m.eval()
            # print()
    elif args.mode == 'rl':
        m.load_model()
        m.reinforce_tune()

    elif args.mode == 'interact':
        m.load_model()
        m.interactive()
    elif args.mode == 'vocab':
        m.load_glove_embedding()


if __name__ == '__main__':
    main()
