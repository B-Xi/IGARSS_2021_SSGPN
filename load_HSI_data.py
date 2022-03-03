import numpy as np
import scipy.io as sio
import keras
from scipy import sparse
from utils import eucliDist2, normalize
from sklearn import preprocessing
import time

def data_load_test(alpha,w_coef,scl=5,K=20,path=u'data/', dataset='',rate=''):
    if rate=='30':
        x = sio.loadmat(path+dataset+'/'+dataset+'_gyh.mat')[dataset+'_gyh']
        gt=sio.loadmat(path+dataset+'/'+dataset+'_gt.mat')[dataset+'_gt']
        trpos=sio.loadmat(path+dataset+'/trpos.mat')['trpos']-1
        tepos=sio.loadmat(path+dataset+'/tepos.mat')['tepos']-1
        trpos=trpos.astype('int32')
        tepos=tepos.astype('int32')
    else:
        x = sio.loadmat(path+dataset+'/'+dataset+'_gyh.mat')[dataset+'_gyh']
        trmap=sio.loadmat(path+dataset+'/trainingMap.mat')['trainingMap']
        temap=sio.loadmat(path+dataset+'/testingMap.mat')['testingMap']
        gt=trmap+temap
        trpos=np.argwhere(trmap)
        tepos=np.argwhere(temap)
    
    
    if gt[trpos[:,0],trpos[:,1]].min()==0:
        print("wrong\n")
    gt=gt.astype('int')
    row,col,band=x.shape
    num_nodes = trpos.shape[0] + tepos.shape[0]
    num_classes = gt.max()
    train_num=trpos.shape[0]
    
    #flatten
    trpos1D=trpos[:,0]*col+trpos[:,1]
    tepos1D=tepos[:,0]*col+tepos[:,1]
    x = x.reshape(row*col, band)
    gt1D = gt.reshape(row*col)

    # normalize
    x = x.astype(float)
    x = normalize(x, axis=0)
    x[x==0]=0.00000001

    # delete 0 element, get x_all、y_train、y_test、y_all
    pos1D0=np.hstack([trpos1D,tepos1D])
    x_all0=x[pos1D0]
    y_train0=gt1D[trpos1D]-1
    y_test0=gt1D[tepos1D]-1
    y_all0=np.hstack([y_train0,y_test0])
    
    # random rank the generated data
    # 1：trpos1D.shape[0] is the original train_idx
    # trpos1D.shape[0]:end is the original test_idx
    orig_idx=np.arange(num_nodes)
    np.random.shuffle(orig_idx)
    sio.savemat(path+dataset+'/orig_idx'+rate+'.mat', {"orig_idx": orig_idx})
    orig_idx = sio.loadmat(path+dataset+'/orig_idx'+rate+'.mat')['orig_idx']
    orig_idx = np.squeeze(orig_idx,0)
    x_all=np.zeros_like(x_all0)
    x_all[orig_idx]=x_all0
    pos1D=np.zeros_like(pos1D0)
    pos1D[orig_idx]=pos1D0
    y_all=np.zeros_like(y_all0)
    y_all[orig_idx]=y_all0
    
    #get idx_train idx_text train_mask
    train_mask=np.zeros(num_nodes)
    train_mask[orig_idx[0:train_num]]=1
    idx_train=orig_idx[0:train_num]
    idx_test=orig_idx[train_num:]
    
    #onehot encode
    y_train = np.zeros((num_nodes, num_classes))
    y_test = np.zeros((num_nodes, num_classes))
    y_train[idx_train]=keras.utils.to_categorical(y_all[idx_train], num_classes)
    y_test[idx_test]=keras.utils.to_categorical(y_all[idx_test], num_classes)
    
    # Creating Adjacency matrix accordingt to scl, knn
    adj = []
    x = np.arange(scl * scl) - int((scl ** 2 - 1) / 2)
    x = np.round(x / scl).reshape((scl, scl))
    idx_tmp0 = np.array([x.flatten(), x.T.flatten()])
    idx_tmp0 = np.delete(idx_tmp0, int((scl ** 2 - 1) / 2), axis=1) #scl*scl coordinate mask
    ws0=np.exp(-w_coef*(idx_tmp0[0]**2+idx_tmp0[1]**2))
    rcd=0
    data=[]
    indices=[]
    indptr=[0]
    print('\rCreating A matrix ...')
    index_pos1D=np.zeros(row*col)
    for i in range(num_nodes):
        index_pos1D[pos1D[i]]=i
    t = time.time()
    for i in range(num_nodes):
        pos_i = [(int)(pos1D[i] / col), pos1D[i] % col]#重排后非0索引x,y格式
        idx_tmp = np.array([(pos_i[0] + idx_tmp0[0]), pos_i[1] + idx_tmp0[1]])#当前像素的邻域坐标
        #判断邻域坐标是否合法（大于等于0且小于原始像素空间的长/宽）
        idx_mask = np.array(
            (idx_tmp >= 0)[0] * (idx_tmp >= 0)[1] * (idx_tmp[0] < row) * (idx_tmp[1] < col))
        idx_tmp = np.squeeze(idx_tmp[:, np.where(idx_mask)], axis=1)#去除不合法的邻域坐标
        ws=np.squeeze(ws0[idx_mask])
        idx_tmp = np.transpose(idx_tmp)
        idx_tmp1D = idx_tmp[0] * col + idx_tmp[1]#邻域坐标格式由x,y转为1D
        A_i = np.zeros(num_nodes)
        tps=[]
        nei_idx=[]
        for j in range(idx_tmp.shape[0]):
            #查找邻域像素是否在pos1D中，并计算和中心像素的高斯距离
            x1=idx_tmp[j]
            x1D=int(x1[0]*col+x1[1])
            if gt1D[x1D]!=0:
                tp = int(index_pos1D[x1D])
                #tp1 = np.where(pos1D == x1D)[0][0]
                # if tp!=tp1:
                #     print('wrong')
                w=ws[j]
                if alpha==0:
                    alpha=1
                A_i[tp] = np.exp(-alpha*eucliDist2(x_all[i], x_all[tp]))
                tps+=[tp]
                nei_idx+=[j]
        tps=np.array(tps)
        nei_idx=np.array(nei_idx)
        # KNN pixels
        if len(tps)==0:
            print('warning: {:d} of lines is empty!'.format(i))
            indptr+=[indptr[-1]]
        else:
            pos=np.argsort(A_i[tps])#(A[:,i]>0)
            nzero_num=len(tps)
            if nzero_num==0:
                print('\r line {:d} is wrong..'.format(i))
            pos_in_K=pos[-min(K,nzero_num):]
            pos_out_K=pos[-nzero_num:-min(K,nzero_num)]
            A_i[tps[pos_out_K]]=0 #删除K以外边
            K_nearest_nei_idx=nei_idx[pos_in_K] #前K个
            #乘以空间系数w
            if alpha==0:
                A_i[tps[pos_in_K]] =ws[K_nearest_nei_idx]
            else: 
                A_i[tps[pos_in_K]] *=ws[K_nearest_nei_idx]
            indices+=np.sort(tps[pos_in_K]).tolist()
            data+=(A_i[np.sort(tps[pos_in_K])].tolist())
            indptr+=[indptr[-1]+pos_in_K.shape[0]]
        if int(i/num_nodes*100)!=rcd:
            rcd=int(i/num_nodes*100)
            print("\r[{0}{1}]->{2:02d}% {3:.2f}s".format('>'*round(rcd/2),'-'*round((100-rcd)/2),rcd,time.time()-t),end='',flush=True)
    print("\r[{0}{1}]->{2:02d}% {3:.2f}s".format('>'*50,'-'*0,100,time.time()-t),end='\n',flush=True)
    adj=[sparse.csr_matrix((np.array(data),np.array(indices),np.array(indptr)),shape=(num_nodes,num_nodes))]  # 采用行优先的方式压缩矩阵
    sio.savemat(path+dataset+'/AW'+rate+'.mat', {"adj": adj})
    #上述生成第一次完成后，可使用下面的代码进行导入,不用再次计算,不使用如下代码则注释掉即可
    A1 = sio.loadmat(path+dataset+'/AW'+rate+'.mat')['adj']
    if alpha>=0:
        adj_org=A1[0,0]
        adj_org.data = adj_org.data + adj_org.data.T * (adj_org.data.T > adj_org.data) - adj_org.data * (adj_org.data.T > adj_org.data)
        adj = adj_org.tocsr()
    return np.mat(x_all), adj, y_all, y_train, y_test, idx_train, idx_test,pos1D,gt,train_mask.astype(bool)