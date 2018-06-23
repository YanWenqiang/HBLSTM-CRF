import numpy as np 
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import time
from swda_data import load_file
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length

def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length

def minibatches(data, labels, batch_size):
    data_size = len(data)
    start_index = 0

    num_batches_per_epoch = int((len(data) + batch_size - 1) / batch_size)
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index: end_index], labels[start_index: end_index]



hidden_size_lstm_1 = 200
hidden_size_lstm_2 = 200
tags = 39
word_dim = 300
proj1 = 200
proj2 = 100
words = 20001
batchSize = 4
log_dir = "train"
model_dir = "DAModel"
model_name = "ckpt"

class DAModel():
    def __init__(self):
        with tf.variable_scope("placeholder"):

            self.dialogue_lengths = tf.placeholder(tf.int32, shape = [None], name = "dialogue_lengths")
            self.word_ids = tf.placeholder(tf.int32, shape = [None,None,None], name = "word_ids")
            self.utterance_lengths = tf.placeholder(tf.int32, shape = [None, None], name = "utterance_lengths")
            self.labels = tf.placeholder(tf.int32, shape = [None, None], name = "labels")
            self.clip = tf.placeholder(tf.float32, shape = [], name = 'clip')
        
        with tf.variable_scope("embeddings"):
            _word_embeddings = tf.get_variable(
                name = "_word_embeddings",
                dtype = tf.float32,
                shape = [words, word_dim],
                initializer = tf.random_uniform_initializer()
                )
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,self.word_ids, name="word_embeddings")
            self.word_embeddings = tf.nn.dropout(word_embeddings, 0.8)
                    
        with tf.variable_scope("utterance_encoder"):
            s = tf.shape(self.word_embeddings)
            batch_size = s[0] * s[1]
            
            time_step = s[-2]
           
            word_embeddings = tf.reshape(self.word_embeddings, [batch_size, time_step, word_dim])
            length = tf.reshape(self.utterance_lengths, [batch_size])

            fw = BasicLSTMCell(hidden_size_lstm_1, forget_bias=0.8, state_is_tuple= True)
            bw = BasicLSTMCell(hidden_size_lstm_1, forget_bias=0.8, state_is_tuple= True)
            
            (output_fw, output_bw), ((fc, fh), (bc, bh)) = tf.nn.bidirectional_dynamic_rnn(fw, bw, word_embeddings,sequence_length=length, dtype = tf.float32)
            output = tf.concat([output_fw, output_bw], axis = -1)
            
            output_ta = tf.TensorArray(dtype = tf.float32, size = 1, dynamic_size = True)
            
            def body(time, output_ta_1):
                if length[time] == 0:
                    output_ta_1 = output_ta_1.write(time, output[time][0])
                else:
                    output_ta_1 = output_ta_1.write(time, output[time][length[time] - 1])
                return time + 1, output_ta_1

            def condition(time, output_ta_1):
                return time < batch_size

            i = 0
            [time, output_ta] = tf.while_loop(condition, body, loop_vars = [i, output_ta])
            output = output_ta.stack()

            output = tf.reshape(output, [s[0], s[1], 2*hidden_size_lstm_1])
            output = tf.nn.dropout(output, 0.8)
        
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.BasicLSTMCell(hidden_size_lstm_2, state_is_tuple = True)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(hidden_size_lstm_2, state_is_tuple = True)
            
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output, sequence_length = self.dialogue_lengths, dtype = tf.float32)
            outputs = tf.concat([output_fw, output_bw], axis = -1)
            outputs = tf.nn.dropout(outputs, 0.8)
        
        with tf.variable_scope("proj1"):
            output = tf.reshape(outputs, [-1, 2 * hidden_size_lstm_2])
            W = tf.get_variable("W", dtype = tf.float32, shape = [2 * hidden_size_lstm_2, proj1], initializer= tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", dtype = tf.float32, shape = [proj1], initializer=tf.zeros_initializer())
            output = tf.nn.relu(tf.matmul(output, W) + b)

        with tf.variable_scope("proj2"):
            W = tf.get_variable("W", dtype = tf.float32, shape = [proj1, proj2], initializer= tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", dtype = tf.float32, shape = [proj2], initializer=tf.zeros_initializer())
            output = tf.nn.relu(tf.matmul(output, W) + b)


        with tf.variable_scope("logits"):
            nstep = tf.shape(outputs)[1]
            W = tf.get_variable("W", dtype = tf.float32,shape=[proj2, tags], initializer = tf.random_uniform_initializer())
            b = tf.get_variable("b", dtype = tf.float32,shape = [tags],initializer=tf.zeros_initializer())

            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nstep, tags])
        
        with tf.variable_scope("loss"):
            log_likelihood, self.trans_params = tf.contrib.crf.crf_log_likelihood(
                        self.logits, self.labels, self.dialogue_lengths)
            self.loss = tf.reduce_mean(-log_likelihood) + tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
            #tf.summary.scalar("loss", self.loss)
        

        with tf.variable_scope("viterbi_decode"):
            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(self.logits, self.trans_params,  self.dialogue_lengths)
            

            batch_size = tf.shape(self.dialogue_lengths)[0]

            output_ta = tf.TensorArray(dtype = tf.float32, size = 1, dynamic_size = True)
            def body(time, output_ta_1):
                length = self.dialogue_lengths[time]
                vcode = viterbi_sequence[time][:length]
                true_labs = self.labels[time][:length]
                accurate = tf.reduce_sum(tf.cast(tf.equal(vcode, true_labs), tf.float32))

                output_ta_1 = output_ta_1.write(time, accurate)
                return time + 1, output_ta_1


            def condition(time, output_ta_1):
                return time < batch_size

            i = 0
            [time, output_ta] = tf.while_loop(condition, body, loop_vars = [i, output_ta])
            output_ta = output_ta.stack()
            accuracy = tf.reduce_sum(output_ta)
            self.accuracy = accuracy / tf.reduce_sum(tf.cast(self.dialogue_lengths, tf.float32))
            #tf.summary.scalar("accuracy", self.accuracy)



        with tf.variable_scope("train_op"):
            optimizer = tf.train.AdagradOptimizer(0.1)
            #if tf.greater(self.clip , 0):
            grads, vs = zip(*optimizer.compute_gradients(self.loss))
            grads, gnorm = tf.clip_by_global_norm(grads, self.clip)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))
            #else:
            #    self.train_op = optimizer.minimize(self.loss)
        #self.merged = tf.summary.merge_all()
    
def main():
    data, labels = load_file()
    train_data = data[:900]
    train_labels = labels[:900]
    dev_data = data[900:]
    dev_labels = data[900:]
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    
    with tf.Session(config = config) as sess:
        model = DAModel()
        sess.run(tf.global_variables_initializer())
        clip = 2
        saver = tf.train.Saver()
        #writer = tf.summary.FileWriter("D:\\Experimemts\\tensorflow\\DA\\train", sess.graph)
        writer = tf.summary.FileWriter("train", sess.graph)
        counter = 0
        for epoch in range(100):
            
            #dialogues = [[[1,2,3,4],[1,2,3],[2,3,5]],[[1,0], [4]],[[1,2,8,4],[1,1,3],[2,3,9,1,3,1,9]], [[1,2,3,4,5,7,8,9],[9,1,2,4],[8,9,0,1,2]],[[1,2,4,3,2,3],[9,8,7,5,5,5,5,5,5,5,5]],[[1,2,3,4,5,6,9],[9,1,0,0,2,4,6,5,4]],[[1,2,3,4,5,6,7,8,9],[9,1,2,4],[8,9,0,1,2]],[[1]] , [[1,2,11,2,3,2,1,1,3,4,4], [6,5,3,2,1,1,4,5,6,7], [9,8,1], [1,6,4,3,5,7,8], [0,9,2,4,6,2,4,6], [5,2,2,5,6,7,3,7,2,2,1], [0,0,0,1,2,7,5,3,7,5,3,6], [1,3,6,6,3,3,3,5,6,7,2,4,2,1], [1,2,4,5,2,3,1,5,1,1,2], [9,0,1,0,0,1,3,3,5,3,2], [0,9,2,3,0,2,1,5,5,6], [9,0,0,1,4,2,4,10,13,11,12], [0,0,1,2,3,0,1,1,0,1,2], [0,0,1,3,1,12,13,3,12,3], [0,9,1,2,3,4,1,3,2]]]
            #labs = [[1,2,1],[0, 3],[1,2,1],[1,0,2], [2,1], [1,1], [2,1,2], [4], [0,1,2,0,2,4,2,1,0,1,0,2,1,2,0]]
            for dialogues, labels in minibatches(train_data, train_labels, batchSize):
                _, dialogue_lengthss = pad_sequences(dialogues, 0)
                word_idss, utterance_lengthss = pad_sequences(dialogues, 0, nlevels = 2)
                true_labs = labels
                labs_t, _ = pad_sequences(true_labs, 0)
                counter += 1
                train_loss, train_accuracy, _ = sess.run([model.loss, model.accuracy,model.train_op], feed_dict = {model.word_ids: word_idss, model.utterance_lengths: utterance_lengthss, model.dialogue_lengths: dialogue_lengthss, model.labels:labs_t, model.clip :clip} )
                #writer.add_summary(summary, global_step = counter)
                print("step = {}, train_loss = {}, train_accuracy = {}".format(counter, train_loss, train_accuracy))
                
                train_precision_summ = tf.Summary()
                train_precision_summ.value.add(
                    tag='train_accuracy', simple_value=train_accuracy)
                writer.add_summary(train_precision_summ, counter)

                train_loss_summ = tf.Summary()
                train_loss_summ.value.add(
                    tag='train_loss', simple_value=train_loss)
                writer.add_summary(train_loss_summ, counter)
                
                if counter % 1000 == 0:
                    loss_dev = []
                    acc_dev = []
                    for dialogues, labels in minibatches(dev_data, dev_labels, batchSize):
                        _, dialogue_lengthss = pad_sequences(dev_dialogues, 0)
                        word_idss, utterance_lengthss = pad_sequences(dev_dialogues, 0, nlevels = 2)
                        true_labs = dev_labels
                        labs_t, _ = pad_sequences(true_labs, 0)
                        dev_loss, dev_accuacy = sess.run([model.loss, model.accuracy], feed_dict = {model.word_ids: word_idss, model.utterance_lengths: utterance_lengthss, model.dialogue_lengths: dialogue_lengthss, model.labels:labs_t})
                        loss_dev.append(dev_loss)
                        acc_dev.append(dev_accuacy)
                    valid_loss = sum(loss_dev) / len(loss_dev)
                    valid_accuracy = sum(acc_dev) / len(acc_dev)


                    dev_precision_summ = tf.Summary()
                    dev_precision_summ.value.add(
                        tag='dev_accuracy', simple_value=valid_accuracy)
                    writer.add_summary(dev_precision_summ, counter)

                    dev_loss_summ = tf.Summary()
                    dev_loss_summ.value.add(
                        tag='dev_loss', simple_value=valid_loss)
                    writer.add_summary(dev_loss_summ, counter)
                    print("counter = {}, dev_loss = {}, dev_accuacy = {}".format(counter, valid_loss, valid_accuracy))
                
if __name__ == "__main__":
    main()
