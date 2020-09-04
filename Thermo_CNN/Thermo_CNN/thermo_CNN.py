import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.metrics import r2_score
import os
import argparse


#parser = argparse.ArgumentParser('Thermodynamic network to deciper sgRNA activity from sequence feature')
#parser.add_argument('-i' ,'--seq_train', help='One-hot encoded sequence vector', required=True)
#parser.add_argument('-f', '--score', help='Fitness score of sequences', required=True)
#parser.add_argument('-n', '--nn', help='Trainable nearest neighbor', required=True)
#parser.add_argument('-p', '--parameter', help='Thermodynamic parameters', required=True)
#args = parser.parse_args()
seq_train, seq_test = pd.read_pickle('seq_train.pickle'), pd.read_pickle('seq_test.pickle')
score_train, score_test  = pd.read_pickle('score_train.pickle'), pd.read_pickle('score_test.pickle')
#seq, score = pd.read_pickle('seq.pickle'), pd.read_pickle('score.pickle')
BATCH_SIZE = 64
BUFFER_SIZE = 100000
EPOCH = 100
#seq, score = tf.cast(tf.constant(seq), tf.float64), tf.cast(tf.constant(score), tf.float64)
seq_train, seq_test = tf.cast(tf.constant(seq_train), tf.float64), tf.cast(tf.constant(seq_test), tf.float64)
score_train, score_test = tf.cast(tf.constant(score_train), tf.float64), tf.cast(tf.constant(score_test), tf.float64)
indices = tf.range(seq_train.shape[0])
train_indices = tf.data.Dataset.from_tensor_slices(indices).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
RNA_lst = ['A', 'C', 'G', 'U']
DNA_lst = ['A', 'C', 'G', 'T']

# kernel of nearest neighbor
def get_nearest_neighbor():
    nn_lst = ['rAA/dAA-dAT/dTA']
    layer = tf.one_hot(indices=[[[0, 0], [0, 0], [0, 3]]], depth=4, dtype=tf.float64)
    m = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    for n in range(4):
                        if m:
                            conv = tf.one_hot(indices=[[[i, j], [k, l], [n, 3 - k]]], depth=4, dtype=tf.float64)
                            layer = tf.concat([layer, conv], axis=0)
                            nn_lst.append('r'+RNA_lst[i]+RNA_lst[j] +'/d'+DNA_lst[k]+DNA_lst[l]+'-d'+DNA_lst[n]+DNA_lst[3-k]+'/d'+DNA_lst[3-n]+DNA_lst[k])
                        else:
                            m = 1
    return tf.reshape(layer, [-1, 3, 2, 4, 1]), nn_lst

# kernel of initial site
def get_initial_state():
    initial_lst = ['rA-dA-dT']
    layer = tf.one_hot(indices=[[0], [0], [3]], depth=4, dtype=tf.float64)
    m = 0
    for i in range(4):
        for j in range(4):
            if m:
                conv = tf.one_hot(indices=[[i], [j], [3-j]], depth=4, dtype=tf.float64)
                layer = tf.concat([layer, conv], axis=0)
                initial_lst.append('r%s-d%s-d%s'%(RNA_lst[i], DNA_lst[j], DNA_lst[3-j]))
            else:
                m = 1
    return tf.reshape(layer, [-1, 3, 1, 4, 1]), initial_lst

# thermodynamic network
class thermo_net(tf.keras.Model):
    def __init__(self, nearest_neighbor, initial_seq):
        self.nearest_neighbor = nearest_neighbor
        self.initial_seq = initial_seq
        self.ddG1 = tf.Variable(tf.random.normal([1, 1, 1024, 1], dtype=tf.float64), trainable=True)
        self.ddG2 = tf.Variable(tf.random.normal([1, 1, 16, 1], dtype=tf.float64), trainable=True)
        self.w = tf.Variable(tf.random.normal([1], dtype=tf.float64), trainable=True, constraint=lambda x: tf.clip_by_value(x, -10, 0))
        self.b = tf.Variable(tf.random.normal([1], dtype=tf.float64), trainable=True)
    def __call__(self, x):
        nn = tf.concat([tf.nn.conv2d(x[:, :, :-1, :], conv, strides=[1, 1, 1, 1], padding='VALID') for conv in self.nearest_neighbor], axis=-1)
        nn = tf.nn.relu(nn - 5)
        init = tf.concat([tf.nn.conv2d(tf.reshape(x[:, :, -1, :], [-1, 3, 1, 4]), conv, strides=[1, 1, 1, 1], padding='VALID') for conv in self.initial_seq], axis=-1)
        init = tf.nn.relu(init - 2)
        ddG1 = tf.nn.conv2d(nn, self.ddG1, strides=[1, 1, 1, 1], padding='VALID')
        ddG2 = tf.nn.conv2d(init, self.ddG2, strides=[1, 1, 1, 1], padding='VALID')
        ddG = tf.concat([ddG1, ddG2], axis=2)
        ddG = tf.math.cumsum(ddG, axis=2, reverse=True)
        p = tf.nn.softmax(ddG, axis=2)
        p = tf.math.log(p[: ,: ,-1 ,:])
        out = p * self.w + self.b
        return out

nearest_neighbor, nn_lst = get_nearest_neighbor()
initial_seq, initial_lst = get_initial_state()
net = thermo_net(nearest_neighbor, initial_seq)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.2)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer ,thermo_net=net)

def train_step(seq_, score_):
    with tf.GradientTape() as tape:
        score_pred = net(seq_)
        loss = tf.math.reduce_mean(tf.keras.losses.MSE(score_, score_pred))
        gradients = tape.gradient(loss, [net.ddG1, net.ddG2, net.w, net.b])
        optimizer.apply_gradients(zip(gradients, [net.ddG1, net.ddG2, net.w, net.b]))

def train(indices, epochs, seq, score):
    for epoch in range(epochs):
        for data_indices in indices:
            seq_ = tf.gather(seq, data_indices)
            score_ = tf.gather(score, data_indices)
            loss = train_step(seq_, score_)
        if (epoch + 1) % 50 == 0:
        	checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch: %d, loss: %.4f'%(epoch, tf.math.reduce_mean(tf.keras.losses.MSE(score, net(seq)))))

train(train_indices, EPOCH, seq_train, score_train)
parameter = tf.squeeze(net.ddG1).numpy().tolist() + tf.squeeze(net.ddG2).numpy().tolist() + net.w.numpy().tolist() + net.b.numpy().tolist()
parameter_all = pd.DataFrame(parameter, index=nn_lst+initial_lst+['w','b'], columns=['parameter'])
parameter_all.to_csv('parameter_all.csv')
index_trained = pd.read_pickle('nn.pickle')
parameter_trained = parameter_all.loc[index_trained]
parameter_trained.to_csv('parameter_trained.txt')

pairs = pd.read_csv('pairs.txt', sep='\s+')
pairs['dG'] = pairs['dH'] - 303.15 * pairs['dS']
pairs.index = pairs['sequence']
for i in parameter_trained.index[:-3]:
	if i[:7] in pairs.index:
		rd = i[:7]
		dd = i[-7:]
		parameter_trained.loc[i, 'boltzmann'] = (pairs.loc[dd,'dG'] - pairs.loc[rd,'dG']) / (8.314 * 303.15)
nn_check = parameter_trained.dropna()
nn_check.to_csv('nn_check.txt')

x_value = tf.squeeze(net(seq_test)).numpy()
y_value = tf.squeeze(score_test).numpy()
fig, ax = plt.subplots()
ax.scatter(x_value,y_value)
ax.set_xlabel('Model prediction', fontsize='x-large')
ax.set_ylabel('Experimental observation', fontsize='x-large')
xy_min = min(np.min(x_value),np.min(y_value))
xy_max = max(np.max(x_value),np.max(y_value))
plt.xlim((xy_min, xy_max))
plt.ylim((xy_min, xy_max))
r2 = r2_score(y_value, x_value)
#pcc, _ = pearsonr(x_value, y_value)k                                                                                                                                                                                       k
#plt.annotate('Pearson Correlation Coefficient = %.4f'%pcc, xy=(0.05, 0.95), xycoords='axes fraction',fontsize='large')
plt.annotate('$R^2 = %.4f$'%r2, xy=(0.05, 0.9), xycoords='axes fraction',fontsize='x-large')
plt.savefig('Model_prediction.png',dpi=400)
plt.close(fig)

fig,ax = plt.subplots()
ax.plot(nn_check['parameter'], nn_check['boltzmann'], 'k.')
ax.set_xlabel('Parameter', fontsize='x-large')
ax.set_ylabel('-$\Delta \Delta$G/RT', fontsize='x-large')
#plt.subplots_adjust(left=0.15, right=0.8,  bottom=0.1, top=0.9)
scc, _ = spearmanr(nn_check['parameter'], nn_check['boltzmann'])
plt.annotate('Spearman Correlation Coefficient = %.4f'%scc, xy=(0.05, 0.9), xycoords='axes fraction',fontsize='x-large')
plt.savefig('Parameter.png',dpi=400)
plt.close(fig)
