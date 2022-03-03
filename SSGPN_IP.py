from __future__ import print_function
import scipy.io as sio
import keras
from keras.layers import Input,Dense, Dropout, Lambda, Conv1D, concatenate, UpSampling1D, Flatten, AveragePooling1D,GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam
from layers.centers import Centers
from layers.graph import GraphConvolution
from utils import *
from load_HSI_data import data_load_test
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.engine.topology import Layer
from keras import activations,regularizers
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
#-------------------------------------------------------------------------
# Define parameters
#ID=1:Pavia University
#ID=2:Indian Pines
#ID=6:KSC
#ID=7:Houston
datasets=['','paviaU','IP','','','','KSC','HU2012']
ID=2
FILTER = 'chebyshev'  # 'chebyshev' 'localpool'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
#-------------------------------------------------------------------------
# Model define
NB_EPOCH = 500
rate = '_SSGPN'
k=40
scale = 7  # Default size of scale
tau=0.1
w_coef=0.5
####
DCE_FLAG = True # use DCE_loss (True) vs. Cross_entropy_loss (False)
weight_dis = 1.0
beta = 10
T = 1.0
#-------------------------------------------------------------------------
# Random seed set
seed = 123
np.random.seed(seed)
tf.random.set_random_seed(seed)
#-------------------------------------------------------------------------

# Get data
X, A, y, y_train, y_test, idx_train, idx_test, pos1D, gt, train_mask = data_load_test(tau,w_coef,K=k,scl=scale,dataset=datasets[ID],rate=rate)#KSC:0.2
classNum = int(y.max()+1)
idx_train, idx_val = get_idx_train_val(y_train[idx_train], idx_train, classNum, 0.1)  # get idx_train、idx_val
y_val = np.zeros((y_train.shape[0], classNum))
y_val[idx_val] = y_train[idx_val]
y_train[idx_val] = np.zeros((idx_val.shape[0], classNum))
train_mask[idx_val] = False
# Normalize X
X /= X.sum(1).reshape(-1, 1) # delete if pavia
record=np.zeros((NB_EPOCH,4))
#-------------------------------------------------------------------------
# build model's inputs
if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    G = []
    graph = []
    support = 1
    graph = [preprocess_adj(A, SYM_NORM)]
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    G = []
    graph = []
    support = MAX_DEGREE + 1
    L = normalized_laplacian(A, SYM_NORM)
    L_scaled = rescale_laplacian(L)
    graph = chebyshev_polynomial(L_scaled, MAX_DEGREE)
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]
else:
    raise Exception('Invalid filter type.')

X_in = Input(shape=(X.shape[1],))
#-------------------------------------------------------------------------
# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
# def GCN(X_in):
#     #H = Dropout(0.5)(X_in)
#     H = GraphConvolution(25, support, activation='relu')([X_in] + G)#KSC:25
#     #H = Dropout(0.1)(H)#参数可调#KSC:无
#     Y =[GraphConvolution(classNum, support, activation='softmax')([H] + G)]
#     return Y

def SSGPN(X_in, DCE_FLAG):
    units1 = 32
    units2 = classNum
    if DCE_FLAG:
        units2 = 16
    H = GraphConvolution(units1, support, normalization=True, activation='relu')([X_in] + G)
    if DCE_FLAG:
        Encg = GraphConvolution(units2, support, activation='relu', name='APL')([H] + G)
        Encg = Dropout(0.1)(Encg)
        output = Centers(units=classNum, T=T, name='logits', activation='softmax')(Encg)
    else:
        Encg = GraphConvolution(units2, support, name='APL')([H] + G)
        output = Lambda(lambda x: tf.nn.softmax(x))(Encg)
    return output


# -------------------------------------------------------------------------
# the Model output
output = SSGPN(X_in, DCE_FLAG)
y_train=np.argmax(y_train,1)
y_test=np.argmax(y_test,1)
y_val=np.argmax(y_val,1)

# Compile model
model = Model(inputs=[X_in]+G, outputs=([output,output]))
model.summary()
model.compile(loss=(['sparse_categorical_crossentropy',dis_loss]),loss_weights=([1.0, weight_dis]), optimizer=Adam(lr=0.01))

preds = None
print("-----------------------------------")
print("Train samples: {:d}".format(idx_train.shape[0]),
      "\nValidate samples: {:d}".format(idx_val.shape[0]),
      "\nTest samples: {:d}".format(idx_test.shape[0]))
print("-----------------------------------")
# Fit
resultpath = 'data/'+datasets[ID]+'/result/'
if  not os.path.exists(resultpath):
    os.makedirs(resultpath)
f1=open(resultpath+'train_log'+rate+'.txt','w')
train_t = time.time()
#model.load_weights(resultpath+'/model_weights'+rate+'.h5')
for epoch in range(1, NB_EPOCH+1):
    # Log wall-clock time
    t = time.time()
    sample_weights = logsig((np.ones(y_train.shape[0], dtype='float32') - 1 + epoch-1 - NB_EPOCH / 2) / NB_EPOCH * beta)
    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit([X] + graph, [y_train ,y_train], sample_weight=([train_mask,sample_weights]),#, np.ones((y_train.shape[0]))
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)
    # Predict on full dataset
    preds,_ = model.predict([X]+graph, batch_size=A.shape[0])
    # Train / validation scores
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val], epoch - 1, NB_EPOCH, weight_dis)
    record[epoch-1]=np.hstack([train_val_loss,train_val_acc])
    f1.write(str("Epoch: {:04d}".format(epoch)+
              " train_loss= {:.4f} ".format(train_val_loss[0])+
              " train_acc= {:.4f} ".format(train_val_acc[0])+
              " val_loss= {:.4f} ".format(train_val_loss[1])+
              " val_acc= {:.4f} ".format(train_val_acc[1])+
              " time= {:.4f} ".format(time.time() - t))+'\n')
    #if epoch % 50 == 0:
    print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_val_loss[0]),
              "train_acc= {:.4f}".format(train_val_acc[0]),
              "val_loss= {:.4f}".format(train_val_loss[1]),
              "val_acc= {:.4f}".format(train_val_acc[1]),
              "time= {:.4f}".format(time.time() - t))
    if epoch % 100 == 0:
        test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test], NB_EPOCH - 1, NB_EPOCH, weight_dis)
        print("Epoch: {:04d}".format(epoch),
              "Test set results:",
              "loss= {:.4f}".format(test_loss[0]),
              "accuracy= {:.4f}".format(test_acc[0]))
        f1.write(str("Epoch: {:04d} ".format(epoch)+
                      " Test set results:"+
                      " loss= {:.4f} ".format(test_loss[0])+
                      " accuracy= {:.4f} ".format(test_acc[0])))
    if epoch % NB_EPOCH == 0:
        model.save_weights(resultpath + 'model_weights' + rate + '.h5')
f1.close()


#plot and save loss curves
plt.plot(np.arange(NB_EPOCH),record[:,0], label='train_loss')
plt.plot(np.arange(NB_EPOCH), record[:,1], label='val_loss')
plt.legend()
plt.savefig(resultpath+'loss'+'_'+rate+'.png')

#plot and save acc curves
plt.plot(np.arange(NB_EPOCH),record[:,2], label='train_acc')
plt.plot(np.arange(NB_EPOCH), record[:,3], label='val_acc')
plt.legend()
plt.savefig(resultpath+'acc'+'_'+rate+'.png')

# model.save('model.h5')
model.load_weights(resultpath+'model_weights'+rate+'.h5')
training_time = time.time() - train_t
print("Training time =", str(time.time() - train_t))
# Testing
test_t = time.time()
preds,_ = model.predict([X]+graph, batch_size=A.shape[0])
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test], NB_EPOCH-1, NB_EPOCH, weight_dis)
test_time = time.time() - test_t
print("Testing time =", str(time.time() - test_t))

print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))

matrix = np.zeros((classNum, classNum))
data = np.argmax(preds[idx_test], 1)
test_labels = y_test[idx_test]
n = (idx_test.__len__())
with open(resultpath+'prediction'+rate+'.txt', 'w') as f:
    for i in range(n):
        pre_label = int(data[i])
        f.write(str(pre_label)+'\n')
        matrix[pre_label][test_labels[i]] += 1
np.savetxt(resultpath+'result_matrix'+rate+'.txt', matrix, fmt='%d', delimiter=',')
print(''+str(np.int_(matrix)))
print(np.sum(np.trace(matrix)))


# print('OA = '+str(OA)+'\n')
ua = np.diag(matrix)/np.sum(matrix, axis=0)
AA=np.sum(ua)/matrix.shape[0]

precision = np.diag(matrix)/np.sum(matrix, axis=1)
matrix = np.mat(matrix)
OA = np.sum(np.trace(matrix)) / float(n)

Po = OA
xsum = np.sum(matrix, axis=1)
ysum = np.sum(matrix, axis=0)
Pe = float(ysum*xsum)/(np.sum(matrix)**2)
Kappa = float((Po-Pe)/(1-Pe))

AP=np.sum(precision)/matrix.shape[0]

# print('ua =')
for i in range(classNum):
    print(ua[i])
print(AA)
print(OA)
print(Kappa)
print()
for i in range(classNum):
    print(precision[i])
print(AP)
print(training_time)
print(test_time)

f.close()
stat_res=np.hstack([ua,AA,OA,Kappa,precision,AP,training_time,test_time])

pred_map=generate_map(data+1,pos1D[idx_test],gt)
# save results
sio.savemat(resultpath+'pred_map'+rate+'.mat', {'pred_map': pred_map})
sio.savemat(resultpath+'pred'+rate+'.mat', {'pred': preds})
sio.savemat(resultpath+'stat_res'+rate+'.mat', {'stat_res': stat_res})
plt.figure()
img_gt = DrawResult(gt.reshape(pred_map.shape[0],1),ID)
plt.figure()
img = DrawResult(pred_map,ID)
plt.imsave(resultpath+'GCN'+'_'+repr(int(OA*10000))+rate+'.png',img)
plt.show()
