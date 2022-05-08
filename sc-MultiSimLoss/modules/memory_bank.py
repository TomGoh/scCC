'''
Author: Haoze Wu
Date: 2022-04-13 13:41:22
LastEditors: Haoze Wu
LastEditTime: 2022-04-13 13:42:40
FilePath: \scCC\memory_bank.py
Description: 

Copyright (c) 2022 by Haoze Wu, All Rights Reserved. 
'''

import hnswlib
import numpy as np


class MemoryBank():

    # 初始化，传入参数
    def __init__(self,batch_size,full_data,args,topK=10):
        self.topK=topK
        self.batch_size=batch_size
        self.bank=None
        self.full_data=full_data
        self.args=args

    # 根据在updateBank中更新的hnsw对象以及输入的数据data（这里可以是embedding）提取TopK个近邻的数据
    # 返回的结果是一个形状为[TopK,batch_size,num_genes]的数组，从第一个维度来看，
    # 每个[batch_size,num_genes]的子数组都是根据输入的数据data寻找的一个近邻，一共TopK个
    def generateContrast(self,data):
        if self.bank is not None:
            contrasts=np.empty((self.topK,self.args.batch_size,self.args.num_genes))
            labels,distances=self.bank.knn_query(data,k=self.topK)
            
            # print(labels)

            for step,label in enumerate(labels):
                contrasts[:,step]=self.full_data[label.tolist()]
            return contrasts
        else:
            print('Memory Bank has not been initialized......')
            raise NotImplementedError()

    # 根据输入的embedding更新hnsw对象
    def updateBank(self,embedding):
        num_elements=len(embedding)
        dim=embedding.shape[1]
        self.bank=hnswlib.Index(space='cosine',dim=dim)
        self.bank.init_index(max_elements=num_elements, ef_construction=100, M=16)
        self.bank.set_ef(100)
        self.bank.set_num_threads(4)
        self.bank.add_items(embedding)

class StaticMemoryBank_for_MSLOSS():

    def __init__(self,batch_size,x,dim,nn_counts,max_elements):
        self.batch_size=batch_size
        self.dim=dim
        self.nn_counts=nn_counts
        self.bank=hnswlib.Index(space='cosine',dim=dim)
        self.bank.init_index(max_elements=max_elements, ef_construction=100, M=16)
        self.bank.set_ef(100)
        self.bank.set_num_threads(4)
        self.bank.add_items(x)
        self.x_data=x
 
    def generate_data(self,sample):

        labels,distances=self.bank.knn_query(sample,k=self.nn_counts)
        pseudolabel=np.arange(labels.shape[0])
        pseudolabel=np.repeat(pseudolabel,self.nn_counts).reshape(-1)
        
        labels=labels.reshape(-1)
        data=self.x_data[labels]
        
        return data,pseudolabel

class StaticMemoryBank_for_MSLOSS_SelfEnhanced():

    def __init__(self,batch_size,x,dim,nn_counts):
        self.batch_size=batch_size
        self.dim=dim
        self.nn_counts=nn_counts
        self.bank=hnswlib.Index(space='cosine',dim=dim)
        self.bank.init_index(max_elements=8569, ef_construction=100, M=16)
        self.bank.set_ef(100)
        self.bank.set_num_threads(4)
        self.bank.add_items(x)
        self.x_data=x
 
    def generate_data(self,sample):

        labels,distances=self.bank.knn_query(sample,k=self.nn_counts)
        pseudolabel=np.arange(labels.shape[0])
        pseudolabel=np.repeat(pseudolabel,self.nn_counts).reshape(-1)
        
        # print(labels[0])
        self_index=labels[:,0]
        labels[:,-1]=self_index
        labels[:,-2]=self_index
        labels[:,-3]=self_index
        # print(self_index.shape)
        # print(labels.shape)
        # print(labels[0])
        labels=labels.reshape(-1)

        data=self.x_data[labels]

        return data,pseudolabel
    
        
