import numpy as np
import pickle
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from preprocess_dataloader import GeneratorPretrainingGenerator
from postprocess_data_generator import generates_samples
from similarity_generated_seed import save_mapping
from generator import Generator
from rewarder import Rewarder
from rollout_ppo import ROLLOUT
import os
import time
from transformers import pipeline
from transformers import AutoModelForMaskedLM, AutoTokenizer
# from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch
import re
import math
import argparse
import os
#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 128 # embedding dimension
HIDDEN_DIM = 128 # hidden state dimension of lstm cell
START_TOKEN = 0
SEED = 88

#########################################################################################
#  Reward Hyper-parameters
#########################################################################################
MID_LAYER_G = [256]
MID_LAYER_R = [512]
re_dropout_keep_prob = 0.45
re_l2_reg_lambda = 1e-5

ent_w = 1                                          # weight of entropy regulariation term, higher ent_w generates more diverse text (quality tradeoff)
R_decay = 16 # SGD learn epoch decay
R_rate = 0.01

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 30                             # Total number of adversarial training iterations
# file_one_walk = save_dir+'/one_walk_data.txt'           # one-walk text data
# file_seed = save_dir+'/seed.txt'                        # seed text data 
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
output_dir = os.path.abspath(output_dir)

save_dir = "save"
os.makedirs(save_dir, exist_ok=True)
save_dir = os.path.abspath(save_dir)
union_file_path = os.path.join(save_dir,'union_seed_one_walk.txt')   # saves union of one-walk and seed data

positive_file_union = os.path.join(save_dir,'union_training_data.txt')       # training data = converts the union_file_path into ID for pre-training       
# positive_file_seed = os.path.join(save_dir,'seed_training_data.txt')          # training data = converts the file_seed into ID for adversarial training      

# positive_file = positive_file_seed                # GAN training will happen on positive file. If you want to train on Union of Seed/One-walk. then use positive_file_union

negative_file = os.path.join(save_dir,'generator_sample'+str(ent_w)+'.txt')
eval_file_prefix = os.path.join(save_dir,'evaler_file'+str(ent_w))
pretrain_file_prefix = os.path.join(save_dir,'pretrain_file'+str(ent_w))
generated_num = 10000
final_generated_output = 10000
restore = False                                      # restores pre-trained generator and rewarder
off_num = 2048  # off_policy samples(use PPO2)


with open(eval_file_prefix + "_dloss.txt","w",encoding="utf-8") as f:
        f.write(str(0.0))
        
with open("./save/evaler_file1_dloss.txt","w",encoding="utf-8") as f:
        f.write(str(0.0))

def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        samples = trainable_model.generate(sess)
        generated_samples.extend(samples)

    with open(output_file, 'w', encoding="utf-8") as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

def off_policy_samples(sess, trainable_model, batch_size, generated_num):
    off_policy_sample = []
    off_policy_probs = []
    for _ in range(int(generated_num / batch_size)):
        samples, sample_probs = trainable_model.generate(sess)
        off_policy_sample.append(samples)
        off_policy_probs.append(sample_probs)

    return off_policy_sample, off_policy_probs

def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)

def sigmoid(x):
  return 1/(1+np.exp(-x))

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def main(args):


    print("length_query = ", length_query, "seed_times_size = ",seed_times_size, "batch_size_train_value = ",batch_size_train_value, "top_k_bert_value = ",top_k_bert_value, "flag_Bert_MLM = ",flag_Bert_MLM)
#######################################
    PRE_EPOCH_NUM = steps_pre_training # supervise (maximum likelihood estimation) epochs for generator pre-training
    SEQ_LENGTH = length_query # Max sequence length of query that you want to generate (As small as you can)

    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    # with open("Jews_Anti_Sematic_Hate.tsv",'r') as f:
    #     lst = f.read().splitlines()

    with open(file_seed,"r", encoding="utf-8") as training_data:
        seed = training_data.read().splitlines()

    # BATCH_SIZE = int(len(seed)*0.49)     
    # re_batch_size = BATCH_SIZE

    with open(file_one_walk,"r", encoding="utf-8") as vocab_data:
        one_walk = vocab_data.read().splitlines()       # ignore #NAME?
    
    max_len_one_walk_process = min(90000, len(one_walk))
    if max_len_one_walk_process == 90000:
        one_walk = random.sample(one_walk,max_len_one_walk_process)

    union_data = []                             # union data
    
    for i in one_walk:
        if "#NAME?" not in i:
            line=i.encode('utf-8','ignore').decode("ascii", "ignore")
            if line!="":
                union_data.append(i)

    union_data.extend(seed*(int(math.sqrt(len(one_walk)//len(seed)+2))))

    # seed_queries_bert = []
    # with open(file_seed,'r', encoding="utf-8") as file:
    #     for i in file:
    #         seed_queries_bert.append(i.strip())

    ###########################
    
    if flag_multilingual:
        model_name_sbert = 'distiluse-base-multilingual-cased-v1'
    else:
        model_name_sbert = 'bert-base-nli-mean-tokens'
    
    
    with open(union_file_path, "w", encoding="utf-8") as file:
        print("Start Writing One-Walk at :", time.time())
        for i in union_data:
            file.write(i+"\n")
        print("End Writing One-Walk at :", time.time())
#         if flag_Bert_MLM:
#             print("Starting BERT MLM at :", time.time())
#             tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#             model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
#             unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer, top_k = top_k_bert_value, device= 0)  # remove device if on CPU

#             for grouped_sentence in chunker(seed,5):
#                 replacement_word_dic = {}
#                 total_sentence = ' '.join(grouped_sentence)
#                 list_total_sentence = total_sentence.split()
#                 for sentence_idx in range(len(list_total_sentence)):
#                     grouped_sentence_with_mask = list_total_sentence[:]
#                     grouped_sentence_with_mask[sentence_idx] = tokenizer.mask_token
#                     text = ' '.join(grouped_sentence_with_mask)
#                     top_10 = unmasker(text)
#                     for token in top_10:
#                         word = token["token_str"]
#                         if (bool(re.match("^[A-Za-z0-9]*$",word))):
#                             try:
#                                 replacement_word_dic[list_total_sentence[sentence_idx]] += [word]
#                             except:
#                                 replacement_word_dic[list_total_sentence[sentence_idx]] = [word]

#                 for query in grouped_sentence:
#                     for j in query.split():
#                         try:
#                             for k in list(set(replacement_word_dic[j])):
#                                 new_query = query.replace(j,k)
#                                 file.write(new_query+"\n")
#                                 # file_bert.write(new_query+"\n")
#                         except:
#                             pass
    
#     print("Ending BERT MLM at :",time.time())

    
    file_seed_multiplier = file_seed+"multiplier" 
    with open(file_seed_multiplier,"w", encoding="utf-8") as f:
        seed_multiplied = seed*seed_times_size
        for i in range(len(seed_multiplied)):
            if seed_multiplied[i]!="" and i<len(seed_multiplied)-1:
                f.write(seed_multiplied[i]+"\n")
            elif i==len(seed_multiplied)-1:
                f.write(seed_multiplied[i])
        
    # BATCH_SIZE = int(len(seed)*batch_size_train_value)     
    # re_batch_size = BATCH_SIZE
    positive_file = file_seed_multiplier
    positive_file_seed = positive_file
    print("Start converting text data to numbers for input to IRL :", time.time())
    # convert both seed text data and union text data to numbers for input to IRL
    generate_training_data = GeneratorPretrainingGenerator(path_union = union_file_path, path_seed = file_seed_multiplier, positive_file_union = positive_file_union, positive_file_seed = positive_file_seed, T=SEQ_LENGTH)
    print("End converting text data to numbers for input to IRL :", time.time())
    id2word_dict_path = os.path.join(save_dir,"id2word.pkl")
    with open(id2word_dict_path, 'rb') as handle:
        id2word = pickle.load(handle)

    params_path = os.path.join(save_dir,"parameters.txt")
    with open(params_path,"r") as params:
        lines = params.readlines()

    vocab_size = int(lines[0])
    print("vocab_size = ", vocab_size)

    with open(positive_file_seed,"r", encoding="utf-8") as training_data:
        seed_numerical = training_data.read().splitlines()
    # print(len(seed_numerical),"seed_numerical")
    BATCH_SIZE = min(2*int(len(seed_numerical)*batch_size_train_value*1.0/seed_times_size),64)
    re_batch_size = BATCH_SIZE
    print("Final Batch Size = ", BATCH_SIZE)
    tf.compat.v1.disable_eager_execution()
    gen_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)
    dis_data_loader = Dis_dataloader(re_batch_size, SEQ_LENGTH)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, MID_LAYER_G)
    rewarder = Rewarder(vocab_size, BATCH_SIZE, EMB_DIM * 2, HIDDEN_DIM * 2, SEQ_LENGTH, START_TOKEN, MID_LAYER_R, l2_reg_lambda=re_l2_reg_lambda)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    gen_data_loader.create_batches(positive_file_union)             # generator pre-trained on Union of Seed, One-Walk and BERT
    log_path = os.path.join(save_dir,'experiment-log-'+str(ent_w)+'.txt')
    log = open(log_path, 'a')
    #  pre-train generator
    if restore is False:
        print ('Start pre-training...')
        log.write('pre-training...\n')
        for epoch in range(PRE_EPOCH_NUM):
            loss = pre_train_epoch(sess, generator, gen_data_loader)
            if epoch % 20 == 0:
                print ('pre-train epoch ', epoch, 'test_loss ', loss)
                buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(loss) + '\n'
                log.write(buffer)
            if epoch == PRE_EPOCH_NUM-1:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, pretrain_file_prefix + str(epoch))

        print ('Start pre-training rewarder...')
        start = time.time()
        for _ in range(4):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file_union, negative_file)

            for _ in range(1):
                dis_data_loader.reset_pointer()
                r_losses = []
                for it in range(dis_data_loader.num_batch):
                    x_text = dis_data_loader.next_batch()
                    _, r_loss = rewarder.reward_train_step(sess, x_text, np.ones(BATCH_SIZE), 1.0, re_dropout_keep_prob, 0.01)
                    r_losses.append(r_loss)
                log.write('pre-train reward_loss ' +str(np.mean(r_losses))+"\n")
        speed = time.time() - start
        print ('Reward pre_training Speed:{:.3f}'.format(speed))

        checkpoint_path = os.path.join('save', 'exper_40.ckpt')
        saver.save(sess, checkpoint_path)
    else:
        print ('Restore pretrained model ...')
        log.write('Restore pre-trained model...\n')
        ckpt = tf.train.get_checkpoint_state('save')
        saver.restore(sess, ckpt.model_checkpoint_path)
    # by setting the parameters to 0.0 and 1.0, we didn't use the mixed policy RL training in SeqGAN
    rollout = ROLLOUT(generator, 0.0, 1.0)

    print ('#########################################################################')
    print ('Start Adversarial Training...')
    log.write('adversarial training...\n')
    avg_train_loss_g = 0
    avg_train_loss_d = 0
    
    for total_batch in range(TOTAL_BATCH):
        # if (total_batch == 0):      
        #     generate_samples(sess, generator, BATCH_SIZE, final_generated_output, eval_file_prefix + str(total_batch))
        #     output_path = file_output_MLE_PreTraining
        #     generates_samples(g_data_path = eval_file_prefix + str(total_batch), output_file = output_path, id2word = id2word)
        #     # if (total_batch % 3 == 0):
        #     # store_generated_samples_in = os.path.join(output_dir,str(total_batch)+"_"+"results.txt")
        #     save_mapping(output_path, file_seed, file_output_irl_seqGAN)



#         if (total_batch%3==0) or (total_batch > BATCH_SIZE-4):
        
#             if (total_batch > 10):      
        generate_samples(sess, generator, BATCH_SIZE, final_generated_output, eval_file_prefix + str(total_batch))
        output_path = eval_file_prefix + str(total_batch) + ".txt"
        generates_samples(g_data_path = eval_file_prefix + str(total_batch), output_file = output_path, id2word = id2word)
        print("output done for batch --  ",str(total_batch))
        # if (total_batch % 3 == 0):
        # store_generated_samples_in = os.path.join(output_dir,str(total_batch)+"_"+"results.txt")
#         print("Start Sentence-BERT on generated Text :", time.time())
#         save_mapping(output_path, file_seed, file_output_irl_seqGAN, model_name_sbert)
#         print("End Sentence-BERT on generated Text :", time.time())

        # Train the generator for one step
        start = time.time()
        g_losses = []
        off_samples, off_probs = off_policy_samples(sess, rollout, BATCH_SIZE, off_num)
        avg_reward = []
        
        
        file_path_1 =  eval_file_prefix + str(total_batch) + "_gloss.txt"
        file_path_2 =  eval_file_prefix + str(total_batch) + "_dloss.txt"

        while not os.path.exists(file_path_1):
            time.sleep(5)

        if os.path.isfile(file_path_1):
            # read file
            with open(file_path_1,"r",encoding="utf-8") as f:
                avg_train_loss_g=f.read()
        else:
            raise ValueError("%s isn't a file!" % file_path)

        while not os.path.exists(file_path_2):
            time.sleep(15)

        if os.path.isfile(file_path_2):
            # read file
            with open(file_path_2,"r",encoding="utf-8") as f:
                avg_train_loss_d=f.read()
        else:
            raise ValueError("%s isn't a file!" % file_path)
        
        
        for it in range(off_num // BATCH_SIZE):
            rewards = rollout.get_reward(sess, off_samples[it], 4, rewarder)
            avg_reward.append(rewards+float(avg_train_loss_g))
        baseline = np.zeros(SEQ_LENGTH)
#         print(avg_reward)
#         avg_reward=np.array(avg_reward)
#         new_avg_reward = avg_reward[:,0]/max(avg_reward[:,0])+avg_reward[:,1]/max(avg_reward[:,1])
#         new_avg_reward.tolist()
        for it in range(1):
            for it2 in range(off_num // BATCH_SIZE):
                _, g_loss = generator.rl_train_step(sess, off_samples[it2], avg_reward[it2], baseline, off_probs[it2], ent_w)
                g_losses.append(g_loss)
        speed = time.time() - start
        print ('MaxentPolicy Gradient {} round, Speed:{:.3f}, Loss:{:.3f}'.format(total_batch, speed, np.mean(g_losses)))
        print(str(avg_train_loss_g) + "  avg_train_loss_g") 
#         generator_loss_path = os.path.join(save_dir,"generator_training_loss.txt")
#         with open(generator_loss_path,"a", encoding="utf-8") as f:
#             f.write('MaxentPolicy Gradient {} round, Speed:{:.3f}, Loss:{:.3f}'.format(total_batch, speed, np.mean(g_losses))+"\n")

#         with open(eval_file_prefix +"generator_loss_"+str(total_batch) + ".txt", 'w', encoding="utf-8"):
            
        # Update roll-out parameters
        rollout.update_params()

        # Train the rewarder
        start = time.time()
        r_loss_list = []
        for _ in range(8):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            # print(len(negative_file), len(positive_file),BATCH_SIZE,  " --debug output")
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_text= dis_data_loader.next_batch()
                    weights = rewarder.reward_weight(sess, x_text, generator)
                    _, r_loss = rewarder.reward_train_step(sess, x_text, weights, 1, re_dropout_keep_prob, R_rate * np.exp(-(total_batch // R_decay)))
                    r_loss_list.append(r_loss)

        avg_loss = np.mean(r_loss_list)
        speed = time.time() - start
        print ('Reward training {} round, Speed:{:.3f}, Loss:{:.3f}'.format(total_batch, speed, avg_loss))
        rewarder_loss_path = os.path.join(save_dir,"rewarder_training_loss.txt")
        with open(rewarder_loss_path,"a", encoding="utf-8") as fr:
            fr.write('Reward training {} round, Speed:{:.3f}, Loss:{:.3f}'.format(total_batch, speed, avg_loss)+"\n")

#     while(
        
   

    log.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='IRL SeqGAN Query Generation')
    # parser.add_argument('--output_dir', default='results')
    parser.add_argument('--output_IRLseqGAN', default='IRL_SeqGAN_output_Generated_Synthetic_Query.tsv')
    # parser.add_argument('--output_BERT_MLM', default='IRL_SeqGAN_output_BERT_MLM.tsv')
    # parser.add_argument('--output_MLE_PreTraining', default='IRL_SeqGAN_output_MLE_PreTraining.tsv')
    # parser.add_argument('--save_dir', default='save')
    parser.add_argument('--seed_file', type=str)
    parser.add_argument('--one_walk_file', type=str)
    # parser.add_argument('--number_generated_queries', type=int, default=100000)
    parser.add_argument('--size_seed_times', type=int, default=1)
    parser.add_argument('--batch_size_train', type=float, default=1)
    parser.add_argument('--query_length', type=int, default=64)
    parser.add_argument('--pre_train_steps', type=int, default=120)
    parser.add_argument('--top_k_bert', type=int, default=5)
    parser.add_argument('--BERT_MLM_flag', type=bool, default = False)
    parser.add_argument('--multilingual_flag', type=bool, default = False)
    # parser.add_argument('--mode', type=str, choices=['MLM', 'GAN', 'Metric'])
    args = parser.parse_args()
    length_query = args.query_length
    file_seed = args.seed_file
    file_one_walk = args.one_walk_file
    file_output_irl_seqGAN = args.output_IRLseqGAN
    seed_times_size = args.size_seed_times
    batch_size_train_value = args.batch_size_train
    steps_pre_training = args.pre_train_steps
    top_k_bert_value = args.top_k_bert
    flag_Bert_MLM = args.BERT_MLM_flag
    flag_multilingual = args.multilingual_flag
    # file_output_BERT_MLM = args.output_BERT_MLM
    # file_output_MLE_PreTraining = args.output_MLE_PreTraining
    main(args)