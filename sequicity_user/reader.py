import numpy as np
import json
import pickle
from sequicity_user.config import global_config as cfg
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
import random
import os
import re
import csv
import time, datetime
import pdb
import nltk 



def clean_replace(s, r, t, forward=True, backward=False):
    def clean_replace_single(s, r, t, forward, backward, sidx=0):
        idx = s[sidx:].find(r)
        if idx == -1:
            return s, -1
        idx += sidx
        idx_r = idx + len(r)
        if backward:
            while idx > 0 and s[idx - 1]:
                idx -= 1
        elif idx > 0 and s[idx - 1] != ' ':
            return s, -1

        if forward:
            while idx_r < len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
                idx_r += 1
        elif idx_r != len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
            return s, -1
        return s[:idx] + t + s[idx_r:], idx_r

    sidx = 0
    while sidx != -1:
        s, sidx = clean_replace_single(s, r, t, forward, backward, sidx)
    return s


class _ReaderBase:
    class LabelSet:
        def __init__(self):
            self._idx2item = {}
            self._item2idx = {}
            self._freq_dict = {}

        def __len__(self):
            return len(self._idx2item)

        def _absolute_add_item(self, item):
            idx = len(self)
            self._idx2item[idx] = item
            self._item2idx[item] = idx

        def add_item(self, item):
            if item not in self._freq_dict:
                self._freq_dict[item] = 0
            self._freq_dict[item] += 1

        def construct(self, limit):
            l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
            print('Actual label size %d' % (len(l) + len(self._idx2item)))
            if len(l) + len(self._idx2item) < limit:
                logging.warning('actual label set smaller than that configured: {}/{}'
                                .format(len(l) + len(self._idx2item), limit))
            for item in l:
                if item not in self._item2idx:
                    idx = len(self._idx2item)
                    self._idx2item[idx] = item
                    self._item2idx[item] = idx
                    if len(self._idx2item) >= limit:
                        break

        def encode(self, item):
            return self._item2idx[item]

        def decode(self, idx):
            return self._idx2item[idx]

    class Vocab(LabelSet):
        def __init__(self, init=True):
            _ReaderBase.LabelSet.__init__(self)
            if init:
                self._absolute_add_item('<pad>')  # 0
                self._absolute_add_item('<go>')  # 1
                self._absolute_add_item('<unk>')  # 2
                self._absolute_add_item('<go2>')  # 3

        def load_vocab(self, vocab_path):
            f = open(vocab_path, 'rb')
            dic = pickle.load(f)
            self._idx2item = dic['idx2item']
            self._item2idx = dic['item2idx']
            self._freq_dict = dic['freq_dict']
            f.close()

        def save_vocab(self, vocab_path):
            f = open(vocab_path, 'wb')
            dic = {
                'idx2item': self._idx2item,
                'item2idx': self._item2idx,
                'freq_dict': self._freq_dict
            }
            pickle.dump(dic, f)
            f.close()

        def sentence_encode(self, word_list):
            return [self.encode(_) for _ in word_list]

        def sentence_decode(self, index_list, eos=None):
            l = [self.decode(_) for _ in index_list]
            if not eos or eos not in l:
                return ' '.join(l)
            else:
                idx = l.index(eos)
                return ' '.join(l[:idx])

        def nl_decode(self, l, eos=None):
            return [self.sentence_decode(_, eos) + '\n' for _ in l]

        def encode(self, item):
            if item in self._item2idx:
                return self._item2idx[item]
            else:
                return self._item2idx['<unk>']

        def decode(self, idx):
            # pdb.set_trace()
            if int(idx) < len(self):
                return self._idx2item[int(idx)]
            else:
                return 'ITEM_%d' % (idx - cfg.vocab_size)


    def __init__(self):
        self.train, self.dev, self.test = [], [], []
        self.vocab = self.Vocab()
        self.result_file = ''

    def _construct(self, *args):
        """
        load data, construct vocab and store them in self.train/dev/test
        :param args:
        :return:
        """
        raise NotImplementedError('This is an abstract class, bro')

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5: del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        # for k in del_l:
        #    turn_bucket.pop(k)
        return turn_bucket

    def _mark_batch_as_supervised(self, all_batches):
        supervised_num = int(len(all_batches) * cfg.spv_proportion / 100)
        for i, batch in enumerate(all_batches):
            for dial in batch:
                for turn in dial:
                    turn['supervised'] = i < supervised_num
                    if not turn['supervised']:
                        turn['degree'] = [0.] * cfg.degree_size  # unsupervised learning. DB degree should be unknown
        return all_batches

    def _construct_mini_batch(self, data):
        
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            # print(dial)
            # time.sleep(20)
            if len(batch) == cfg.batch_size:
                all_batches.append(batch)
                batch = []
        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        if len(batch) > 0.5 * cfg.batch_size:
            # print("HELLLLLLO")
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        return all_batches

    def _transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def mini_batch_iterator(self, set_name):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        turn_bucket = self._bucket_by_turn(dial)
        # self._shuffle_turn_bucket(turn_bucket)
        all_batches = []
        for k in turn_bucket:
            batches = self._construct_mini_batch(turn_bucket[k])
            all_batches += batches
        self._mark_batch_as_supervised(all_batches)
        # print(len(all_batches))
        # time.sleep(30)
        random.shuffle(all_batches)
        for i, batch in enumerate(all_batches):
            yield self._transpose_batch(batch)

    def wrap_result(self, turn_batch, gen_m, gen_z, eos_syntax=None, prev_z=None):
        """
        wrap generated results
        :param gen_z:
        :param gen_m:
        :param turn_batch: dict of [i_1,i_2,...,i_b] with keys
        :return:
        """
        # print("line 229")
        # reterived_response = ""
        # j_correct_res = ""
        # f = False 
        # db_entity_file2 = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/trainmodel.json','rb')
        # dataList2= json.load(db_entity_file2)
        # print(self.vocab.sentence_decode(gen_m[0], eos='EOS_M'))
        # print(self.vocab.sentence_decode(gen_m[1], eos='EOS_M'))
        results = []
        # print(len(gen_m))
        # time.sleep(30)
        if eos_syntax is None:
            eos_syntax = {'response': 'EOS_M', 'user': 'EOS_U', 'bspan': 'EOS_Z2'}
        batch_size = len(turn_batch['user'])

        # print("-----------------------")
        # print(batch_size)
        # print(len(gen_m))
        print("WRAAAAAP RESULTTTT")
        for i in range(batch_size):
            f = False 
            # print(turn_batch['user'][i])
            # print(gen_m[i])
            entry = {}
            if prev_z is not None:
                src = prev_z[i] + turn_batch['user'][i]
            else:
                src = turn_batch['user'][i]
            for key in turn_batch:
               

                entry[key] = turn_batch[key][i]
               
                if key in eos_syntax:
                    entry[key] = self.vocab.sentence_decode(entry[key], eos=eos_syntax[key])
                    # print(entry[key]) 
                    # print(key)   
                    # if (key == "response"):
                    #     j_correct_res = entry[key]

                    #     for index in range(len(dataList2)):
                    #       if f :
                    #         break 
                    #     for key2 in dataList2[index]:
                    #      j_question = dataList2[index]["qText"]
                    #      if(j_correct_res == j_question):
                    #          f = True 
                    #          break 
            if i < len(gen_m):
                print(gen_m[i])
                print(entry['response'])

                
                # print(i)
                # print(len(gen_m))
                # print("gen_m ==")
                # print(gen_m)
                # entry['generated_response'] = self.vocab.sentence_decode(gen_m[i], eos='EOS_M')
                # print(i)
                entry['generated_response'] = gen_m[i]
                # print(True)
           
            else:
                entry['generated_response'] = ''
            if gen_z:
                entry['generated_bspan'] = self.vocab.sentence_decode(gen_z[i], eos='EOS_Z2')
            else:
                entry['generated_bspan'] = ''
            results.append(entry)
        # time.sleep(50)    
        write_header = False
        if not self.result_file:
            print("yeaaaah")
            self.result_file = open(cfg.result_path, 'w')
            self.result_file.write(str(cfg))
            write_header = True
            # time.sleep(30)

        field = ['dial_id', 'turn_num', 'user', 'generated_bspan', 'bspan', 'generated_response', 'response', 'u_len',
                 'm_len', 'supervised']
        for result in results:
            del_k = []
            for k in result:
                if k not in field:
                    del_k.append(k)
            for k in del_k:
                result.pop(k)
        writer = csv.DictWriter(self.result_file, fieldnames=field)
        if write_header:
            self.result_file.write('START_CSV_SECTION\n')
            # print(True)
            writer.writeheader()
        writer.writerows(results)
        # time.sleep(30)
        # print(results)
        return results

    def db_search(self, constraints):
        raise NotImplementedError('This is an abstract method')

    def db_degree_handler(self, z_samples, *args, **kwargs):
        """
        returns degree of database searching and it may be used to control further decoding.
        One hot vector, indicating the number of entries found: [0, 1, 2, 3, 4, >=5]
        :param z_samples: nested list of B * [T]
        :return: an one-hot control *numpy* control vector
        """
        control_vec = []

        for cons_idx_list in z_samples:
            constraints = set()
            for cons in cons_idx_list:
                if type(cons) is not str:
                    cons = self.vocab.decode(cons)
                if cons == 'EOS_Z1':
                    break
                constraints.add(cons)
            match_result = self.db_search(constraints)
            degree = len(match_result)
            # modified
            # degree = 0
            control_vec.append(self._degree_vec_mapping(degree))
        return np.array(control_vec)

    def _degree_vec_mapping(self, match_num):
        l = [0.] * cfg.degree_size
        l[min(cfg.degree_size - 1, match_num)] = 1.
        return l


class CamRest676Reader(_ReaderBase):
    def __init__(self):
        super().__init__()
        self._construct(cfg.data, cfg.db, cfg.entity)
        self.result_file = ''

 

    def _split_data(self, encoded_data, split):

        total = sum(split)
        dev_thr = len(encoded_data) * split[0] // total
        test_thr = len(encoded_data) * (split[0] + split[1]) // total
        train, dev, test = encoded_data[:dev_thr], encoded_data[dev_thr:test_thr], encoded_data[0:96]
        print(len(train))
        print(len(test))
        # time.sleep(50)
        return train, dev, test

    def _construct(self, data_json_path, db_json_path, entity_json_path):
    
        construct_vocab = False
        if not os.path.isfile(cfg.vocab_path):
            construct_vocab = True
            print('Constructing vocab file...')

        with open(data_json_path) as raw_data_json:
            raw_data = json.loads(raw_data_json.read().lower())
        with open(db_json_path) as db_json:
            db_data = json.loads(db_json.read().lower())
        with open(entity_json_path) as entity_json:
            entity_data = json.loads(entity_json.read().lower())

        self.db = db_data
        self.entity = entity_data
        tokenized_data = self._get_tokenized_data(raw_data, db_data, construct_vocab)
        if construct_vocab:
            self.vocab.construct(cfg.vocab_size)
            self.vocab.save_vocab(cfg.vocab_path)
        else:
            self.vocab.load_vocab(cfg.vocab_path)
        encoded_data = self._get_encoded_data(tokenized_data)
        self.train, self.dev, self.test = self._split_data(encoded_data, cfg.split)
        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)
        raw_data_json.close()
        db_json.close()

    def db_search(self, constraints):
        # print("line 464")
        match_results = []
        for entry in self.db:
            # print(entry.values())
            # pdb.set_trace()
            entry_values = ' '.join(str(entry) for entry in entry.values())
            match = True
            for c in constraints:
                if c not in entry_values:
                    # print(False)
                    match = False
                    break
            if match:
                match_results.append(entry)
        return match_results

class User_Simulator_Reader(CamRest676Reader):
    def __init__(self):
        super().__init__()

    def normalize(self, text):
        def insertSpace(token, text):
            sidx = 0
            while True:
                sidx = text.find(token, sidx)
                if sidx == -1:
                    break
                if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                        re.match('[0-9]', text[sidx + 1]):
                    sidx += 1
                    continue
                if text[sidx - 1] != ' ':
                    text = text[:sidx] + ' ' + text[sidx:]
                    sidx += 1
                if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
                    text = text[:sidx + 1] + ' ' + text[sidx + 1:]
                sidx += 1
            return text

        text = text.lower()

        text = re.sub(r'^\s*|\s*$', '', text)


        text = ' ' + text + ' '
        

     

        return text

    def NLU(self , response): 


        nltk_tokens = nltk.word_tokenize(response)
        intent = ''
        if('summer' or 'round'  in nltk_tokens):
           intent = 'summer'
        elif ('probation' in nltk_tokens):
           intent = 'probation'
        elif ('advising' or 'advisor' in nltk_tokens):
            intent = 'advising'
        elif ('study' in nltk_tokens):
            intent = 'study_advice'    
        else :
         intent = 'guc_guidelines' 
        return intent  

    def single_slot_delex(self, utt, dictionary):
        # print(utt)
        
        for key, val in dictionary:
            utt = (' ' + utt + ' ')
           
            if key in utt:
                v = utt.find(key)
                vv = v + (len(key))
                utt = utt[0:v] + val[:len(val)-1]+'|'+ key +']' + utt[vv:]
                utt = utt[1:-1]
                return utt
         
            

        return utt

    def load_entity(self, response):

        

        # # replace info in db
        # db_entity_file = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/db_entity_file.pkl','rb')
        # db_entity_list = pickle.load(db_entity_file)
        # db_entity_file.close()
        # response = self.normalize(response)
        # db_entity_file = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/goal.json','rb')
        db_entity_file = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/Guc_Dataset_Goal22.json','rb')
        db_entity_list = json.load(db_entity_file)
        db_entity_file.close()
        # response = self.single_slot_delex(response, db_entity_list)

        return response

    def Entity_rep(self, response, goal):
        intent = self.NLU(response)
        response = self.load_entity(response)
        
        slotpat = re.compile('\[.*?\]')
        slots = re.findall(slotpat, response)
        

        for slot in slots:
            [slot_name, slot_val] = slot[1:-1].split('|')          
            response_x = response.replace(slot, slot_val)
            print(response_x)

        return response

    def _get_tokenized_data(self, raw_data, db_data, construct_vocab):
        print("line 828")
        tokenized_data = []

        for dial_id, dial in enumerate(raw_data):
            tokenized_dial = []
            for turn in dial['dial']:
                turn_num = turn['turn']
                constraint = []
                requested = []
                book = []
                recommend = []
                select = []
                goal = []
                c = 0
                for slot in turn['a']['slu']:
                    if slot['act'] == 'inform' :
                        s = slot['slots'][0][0]
                        s = s +" ,"
       
                        requested.extend(word_tokenize(s))
        

                goal.append('EOS_Z0')
                degree = 0 #len(self.db_search(constraint))
                # requested = sorted(requested)
                book = sorted(book)
                constraint.append('EOS_Z1')
                book.append('EOS_Z3')
                recommend.append('EOS_Z4')
                select.append('EOS_Z5')
                requested.append('EOS_Z2')
                user = word_tokenize(turn['b']['sent']) + ['EOS_U']
                response = word_tokenize(self.Entity_rep(turn['a']['transcript'],goal)) + ['EOS_M']
                tokenized_dial.append({
                    'dial_id': dial_id,
                    'turn_num': turn_num,
                    'user': user,
                    'response': response,
                    'book': book,
                    'goal':goal,
                    'select': select,
                    'recommend': recommend,
                    'constraint': constraint,
                    'requested': requested,
                    'degree': degree,
                })
                # pdb.set_trace()
                if construct_vocab:
                    for word in user + response :
                        self.vocab.add_item(word)
            tokenized_data.append(tokenized_dial)
        return tokenized_data

    def _get_encoded_data(self, tokenized_data):
        encoded_data = []
        for dial in tokenized_data:
            encoded_dial = []
            prev_response = []
            for turn in dial:
                user = self.vocab.sentence_encode(turn['user'])
                response = self.vocab.sentence_encode(turn['response'])
                constraint = self.vocab.sentence_encode(turn['constraint'])
                # print(turn['requested'])
                # time.sleep(50)
                requested = self.vocab.sentence_encode(turn['requested'])
                # print(self.vocab.sentence_decode(requested))
                # print(requested)
                # time.sleep(20)
                select = self.vocab.sentence_encode(turn['select'])
                goal = self.vocab.sentence_encode(turn['goal'])
                recommend = self.vocab.sentence_encode(turn['recommend'])
                book = self.vocab.sentence_encode(turn['book'])
                degree = self._degree_vec_mapping(turn['degree'])
                turn_num = turn['turn_num']
                dial_id = turn['dial_id']
                # pdb.set_trace()
                # final input
                encoded_dial.append({
                    'dial_id': dial_id,
                    'turn_num': turn_num,
                    'user': prev_response + user,
                    'response': response,
                    'bspan': requested ,
                    # 'bspan': goal + constraint + book + select + recommend + requested,
                    'goal': goal,
                    'u_len': len(prev_response + user),
                    'm_len': len(response),
                    'degree': degree,
                })
                # modified
                prev_response = response
            encoded_data.append(encoded_dial)
        return encoded_data

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_maxlen = np.max(lengths)
    if maxlen is not None and cfg.truncated:
        maxlen = min(seq_maxlen, maxlen)
    else:
        maxlen = seq_maxlen
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def get_glove_matrix(vocab, initial_embedding_np):
    """
    return a glove embedding matrix
    :param self:
    :param glove_file:
    :param initial_embedding_np:
    :return: np array of [V,E]
    """
    ef = open(cfg.glove_path, 'r')
    cnt = 0
    vec_array = initial_embedding_np
    old_avg = np.average(vec_array)
    old_std = np.std(vec_array)
    vec_array = vec_array.astype(np.float32)
    new_avg, new_std = 0, 0

    for line in ef.readlines():
        line = line.strip().split(' ')
        word, vec = line[0], line[1:]
        vec = np.array(vec, np.float32)
        word_idx = vocab.encode(word)
        if word.lower() in ['unk', '<unk>'] or word_idx != vocab.encode('<unk>'):
            cnt += 1
            vec_array[word_idx] = vec
            new_avg += np.average(vec)
            new_std += np.std(vec)
    new_avg /= cnt
    new_std /= cnt
    ef.close()
    logging.info('%d known embedding. old mean: %f new mean %f, old std %f new std %f' % (cnt, old_avg,
                                                                                          new_avg, old_std, new_std))
    return vec_array


