#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 1 21:18:24 2018

@author: yan
"""

# this script is used to construct an optimal investment portfolio, output: curIvtSet 
# output: curIvtSet is the set of all structures with the investment benefits 
# output: ivt_str is the investment set under budget (you don't need to install gurobi if you only want curIvtSet)

# In[]:
get_ipython().magic(u'matplotlib inline')
from matplotlib import pyplot as plt
import networkx as nx
import pandas as pd
import copy
import time 
import csv

import os
import sys

from collections import Counter

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
file_dir = os.path.dirname(os.path.realpath('_file_'))
if file_dir not in sys.path:
    sys.path.append(file_dir)
file_dir = os.path.abspath(os.path.join('network_lib'))
if file_dir not in sys.path:
    sys.path.append(file_dir)

#Modules developed by TU Delft team for thiss project
from network_lib import network_prep as net_p
from network_lib import network_visualization as net_v
from network_lib import od_prep as od_p
#from network_lib import weighted_betweenness as betw_w

import geopandas as gp
import numpy as np
from simpledbf import Dbf5
from gurobipy import *

import scipy.integrate as integrate
# In[]: Input data

# mozambique input
network = r'./input/MZ_inputs/Road_all_floods.shp'
centroid = r'./input/MZ_inputs/OD_all_MZ_v1.shp'
dbf = Dbf5(r'./input/MZ_inputs/Bridge_all_floods_v1.dbf')

df_structure = dbf.to_dataframe()

# check if the water depth on df_structure is correct: it should be non-decrease from 5 return to 1000 return period
rperiod =['WD_C5','WD_C10','WD_C20','WD_C50','WD_C75','WD_C100','WD_C200','WD_C250','WD_C500','WD_C1000']   
rpTime = [5, 10, 20, 50, 75, 100, 200, 250, 500, 1000] # return period (year)

for i in range(10):
    df_structure[rperiod[i]] = np.nan_to_num(df_structure[rperiod[i]]) # change WD as NAN to 0

wrongInfo=[] # save the row ID of structure with wrong information   
for i in range(len(df_structure)):
    for j in range(9):
        if df_structure[rperiod[j]][i] > df_structure[rperiod[j+1]][i] :
            wrongInfo.append(i)
            
for item in set(wrongInfo):
    for j in range(10):
        df_structure[rperiod[j]][item]=0 

gdf_points, gdf_node_pos, gdf = net_p.prepare_centroids_network(centroid, network)
# save only the graph information as gdf_clean to increase the computational speed 
gdf_clean_0=gdf.iloc[:, 105:114]
gdf_clean_1=pd.concat([gdf_clean_0, gdf.loc[:,'OBJECTID']], axis=1)
gdf_clean = gdf_clean_1.rename(columns={'OBJECTID': 'OBJECT_ID'}) # rename to keep name consistent with Fiji data
          
# Create Networkx MultiGraph object from the GeoDataFrame
G = net_p.gdf_to_simplified_multidigraph(gdf_node_pos, gdf_clean, simplify=False)

# Change the MultiGraph object to Graph object to reduce computation cost 
G_tograph = net_p.multigraph_to_graph(G)

# Observe the properties of the Graph object    
nx.info(G_tograph)

# Take only the largest subgraph which all connected links
len_old = 0
for g in nx.connected_component_subgraphs(G_tograph):
    if len(list(g.edges())) > len_old:
        G1 = g
        len_old = len(list(g.edges()))        
G_sub = G1.copy()

#print('number of disconnected compoents is', nx.number_connected_components(G_sub))
nx.info(G_sub)

# Save the simplified transport network back into GeoDataFrame
gdf_sub = net_p.graph_to_df(G_sub)

# assign the OD to the closest node of the biggest subgraph: 
gdf_points2, gdf_node_pos2, gdf_new=net_p.prepare_newOD(centroid, gdf_sub)
G2_multi = net_p.gdf_to_simplified_multidigraph(gdf_node_pos2, gdf_new, simplify=False)
G2 = net_p.multigraph_to_graph(G2_multi)
gdf2 = net_p.graph_to_df(G2)
allNode = G2.nodes()
allEdge = G2.edges()
od = gdf_points2['Node']

################### traffic flow matrix ####################################################
#read OD demand matrix

import scipy.io
mat = scipy.io.loadmat(r'./input/MZ_inputs/traffic_matrix.mat')
odflow = mat['traffic_matrix'] 
T = odflow[1:,1:]# OD matrix, unit is number of passenger per day
 
# the output of this section is gdf_points2: OD, gdf_node_pos2:nodes of graph, gdf2:edge of graph, G2: graph object 

# In[]: baseline: find the shortest path for each od to minimize the total travel cost; 
# output: 1) baseCost ($): total travel cost between all OD pairs; 2) basePath : the shortest path between all OD pairs   

n=0
basePath = [[[]for i in range(len(od))] for j in range(len(od))]
baseCost=np.zeros((len(od),len(od)))
for i in range(len(od)):    
    for j in range(i+1,len(od)):
        basePath[i][j]=nx.dijkstra_path(G2,od[i],od[j],weight = 'total_cost')
        baseCost[i][j]=nx.dijkstra_path_length(G2,od[i],od[j],weight = 'total_cost')
        
###### Dictionary of shortest path represented by link id
stpID = [[]for i in range(len(od)*(len(od)-1))] # shortest path represented by link ID
dict_linkNode={}    # key: node id as tuple;  value: link id
for i in range(len(gdf2)):
    a = min(gdf2['FNODE_'][i],gdf2['TNODE_'][i])
    b = max(gdf2['FNODE_'][i],gdf2['TNODE_'][i])
    dict_linkNode[(a,b)] = gdf2['OBJECT_ID'][i]
n=0   
for i in range(len(basePath)):
    for j in range(len(basePath[i])):        
        if len(basePath[i][j])>1:
            for k in range(len(basePath[i][j])-1):   
                a = min(basePath[i][j][k],basePath[i][j][k+1])
                b = max(basePath[i][j][k],basePath[i][j][k+1])
                stpID[n].append(dict_linkNode[(a,b)])                
            n+=1
            
# In[]:             
###### build a dictionary to match structure (key) and F,TNODE_ (value); structure type, and index in df_structure
df_structure['StructureT'] = ''
for i in range(len(df_structure)):
    print i
    if 'Bridge' in str(df_structure['Str_Desc'][i]) or 'bridge' in str(df_structure['Str_Desc'][i]):
        df_structure['StructureT'][i] = 'Bridge'                 
    elif 'Culvert' in str(df_structure['Str_Desc'][i]):
        df_structure['StructureT'][i] = 'Culvert'                                         
    else:
        df_structure['StructureT'][i] = 'Crossing'                                         
# In[]
# build a dictionary to match structure (key) and F,TNODE_ (value); structure type, and index in df_structure
dic_strNode={} # structure ID to FNODE_, TNODE_ of the link
dic_strType={} # structure ID to type
dic_strID={} # structure and index in df_structure
dic_linkID={} # link object ID to row number
dic_strLink={} # structure ID to link ID
for i in range(len(df_structure)):
    dic_strID[df_structure['OBJECTID'][i]]=i 
    dic_strType[df_structure['OBJECTID'][i]] = df_structure['StructureT'][i] 
    dic_strLink[df_structure['OBJECTID'][i]] = df_structure['ROADID'][i]
    
    if df_structure['ROADID'][i] in gdf2['OBJECT_ID'].tolist():
        idx = gdf2['OBJECT_ID'][gdf2['OBJECT_ID'] == df_structure['ROADID'][i]].index[0]
        dic_linkID[df_structure['OBJECTID'][i]] = idx
        node = (gdf2['FNODE_'][idx],gdf2['TNODE_'][idx])
        dic_strNode[df_structure['OBJECTID'][i]]=node

# dictionary for link to all structure
dic_linkStr={} # link ID to ID of all structures in the link
for i in range(len(df_structure)):
    item = df_structure['ROADID'][i]
    if item in dic_linkStr:
        dic_linkStr[item].append(df_structure['OBJECTID'][i])#[dic_linkStr[item],df_structure['OBJECTID'][i]]
    else:
        dic_linkStr[item] = [df_structure['OBJECTID'][i]]
        
# In[]
# for each OD pair: given original travel cost & demand, and cost after disruption, compute loss of consumer surplus   
def compute_usrCost(C_0, N_0, C_1,beta=2.9): 
    
    if C_0==0 or N_0==0:
        return 0
    
    N_1 = N_0*np.exp(-beta*np.log(C_1/C_0)) # demand after the travel cost increase (disruption cost)   
    x = np.linspace(N_1, N_0, num=11)
    C = C_0*np.exp(-1/beta*np.log(x/N_0))-C_0
    surplus_loss = np.trapz(C, dx=(N_0-N_1)/10)+N_1*(C_1-C_0)
    return surplus_loss

# In[]:
# update the graph given whether the link item breaks or not    
def updateGraph(G, stru_damage, link_damage, item, linkBreak):
    stru_damage.add(item)
    
    if not dic_strLink[item] in link_damage:
        link_damage.append(dic_strLink[item])
    
    if linkBreak:
        G[dic_strNode[item][0]][dic_strNode[item][1]]['total_cost']=1e10
    else:
        w = G[dic_strNode[item][0]][dic_strNode[item][1]]['total_cost']
        G[dic_strNode[item][0]][dic_strNode[item][1]]['total_cost']=2*w
          
# In[]:
def computeRepairCost(r, RC, repairC, stdLevel, item):
    i = dic_strID[item]
    l = float(df_structure['Over_Lengt'][i]) 
    if np.isnan(l):
        l=10
    b = float(df_structure['Clear_Widt'][i] )
    if np.isnan(b):
        b=3
            
    # fully demange of infrastructure is extra proportion is greater than 1 
    if df_structure[stdLevel][i]==0 or (df_structure[r][i] - df_structure[stdLevel][i])/df_structure[stdLevel][i]>1:
        cost = RC*l*b
    else:                                     
        cost = (df_structure[r][i] - df_structure[stdLevel][i])/df_structure[stdLevel][i]*\
           1*RC*l*b       
    repairC.append(cost) 
    
# In[]:
def computeNewGraphCost(dic_graph, stru_name, G, baseline, iso, days, isoTrip_sum, num, T, disLoss, demandA_B):

    to_allNode = []   
    loc = 0 if stru_name=='Bridge' else 1 if stru_name=='Culvert' else 2

    for j in range(len(od)): 
        to_allNode.append(nx.single_source_dijkstra_path_length(G,od[j],weight = 'total_cost'))  

    cost_disrupt= np.zeros((len(od),len(od)))                 
    for j in range(len(to_allNode)):
        for k in range(len(od)):
            if k>j:
               cost_disrupt[j][k] = to_allNode[j].get(od[k])

    for index, item in np.ndenumerate(cost_disrupt):
        if item>=1e10:
            cost_disrupt[index]=baseline[index]*20 # add a penalty cost, such as the cost as helicopter 
            iso[index] += T[index] * days / 1e6               
            isoTrip_sum[loc] += T[index] * days / 1e6   
                                   
    num[loc] =  np.sum(np.multiply(cost_disrupt>baseline, T)) * days / 1e6   # number of disrupted trips at the duration (million)
    
    ecoLoss = np.zeros((len(od),len(od)))
    
    for j in range(len(od)): # for each OD pair 
        for k in range(j+1,len(od)):
            C_0 = baseCost[j][k]
            N_0 = T[j][k]
            C_1 = cost_disrupt[j][k]
            if N_0 !=0:
                ecoLoss[j][k]= compute_usrCost(C_0, N_0, C_1)  
      
    disLoss[loc] = np.sum(ecoLoss)* days / 1e6 # economic loss at that duration, it includes both disruption and isolation    
    dic_graph[loc] = copy.deepcopy(G)
    # In[]:  
    #  water flood user disruption cost
    # assumption: when a structure in the link is being disrupted, the link is disrupted   
    # structure threshold: bridge 50, culvert 20, crossing 5 return period
    # this function consider the repair duration: if bridge is damaged, ...
    # bridge takes 1 month for bailey bridge construction, with double user cost for the rest of year for bridge construction;
    # culvert takes a month for repair,
    # crossing: a week for repair
    # Input:
    #       r: return period
    #       graph: original traffic network
    #       curIvtSet: links already invested
    #       baseline: cost with no link disruption
    
def disrupt(r,graph,curIvtSet,baseline,bridgeRC_r, culvertRC_r, crossingRC_r, bridge_r, \
            culvert_r, crossing_r,demand,dema_r):
    
    T=demand*dema_r
    bridgeRC = 40000*bridgeRC_r # repair cost $
    culvertRC = 10000*culvertRC_r
    crossingRC = 1000*crossingRC_r    
    bridgeT= 150*bridge_r # repair duration
    culvertT = 24*culvert_r
    crossingT = 7*crossing_r
    
    stru_damage=set()    # ID of the bridge disrupted in that return period
    link_damage=[]    # OBJECT_ID of the road disrupted in that return period:
    repairC1 = [] # total repair cost in that return period (bridge)
    repairC2 = [] # (culvert)
    repairC3 = [] # crossing
    dic_graph = {}    
    bridge_set=set()
    culvert_set=set()
    cross_set=set()
    isoTrip_sum=[0, 0, 0]
    num = [0, 0, 0]
    disLoss=[0, 0, 0]
    demandA_B=[0, 0, 0]
    
    iso = np.zeros((len(baseline), len(baseline)))
    
    G = copy.deepcopy(graph)
    for link, strList in dic_linkStr.items():
        for item in strList:
            if dic_strType[item] == 'Bridge' or dic_strType[item] == 'Footbridge': 
                bridge_set.add(item)
            elif dic_strType[item] == 'Culvert':
                culvert_set.add(item)
            else:
                cross_set.add(item)
                
###################################################################   using bailey bridge, double travel cost 

    for item in bridge_set:
        if dic_strNode.get(item,0)==0: continue
        i = dic_strID[item]
        
        if item in curIvtSet and df_structure[r][i]>df_structure['WD_C100'][i]: # invest but disrupt, assume after invest, design standard=100 yr
            updateGraph(G, stru_damage, link_damage, item, True)
            computeRepairCost(r, bridgeRC, repairC1, 'WD_C100', item)
            
        elif item not in curIvtSet:        
            if df_structure[r][i]>df_structure['WD_C50'][i]:                           
                updateGraph(G, stru_damage, link_damage, item, True)
                computeRepairCost(r, bridgeRC, repairC1, 'WD_C50', item)
    computeNewGraphCost(dic_graph, 'Bridge', G, baseline, iso, bridgeT, isoTrip_sum, num, T, disLoss, demandA_B)   

############################################################## repair culvert and building bailey bridge

    for item in culvert_set:
        if dic_strNode.get(item,0)==0: continue
        i = dic_strID[item]
        
        if item in curIvtSet and df_structure[r][i]>df_structure['WD_C50'][i]: # invest but disrupt, assume after invest, design standard=100 yr
            updateGraph(G, stru_damage, link_damage, item, True)
            computeRepairCost(r, culvertRC, repairC2, 'WD_C50', item)
            
        elif item not in curIvtSet:        
            if df_structure[r][i]>df_structure['WD_C20'][i]:
                updateGraph(G, stru_damage, link_damage, item, True)
                computeRepairCost(r, culvertRC, repairC2, 'WD_C20', item)                                                          

    computeNewGraphCost(dic_graph, 'Culvert', G, baseline, iso, culvertT, isoTrip_sum, num, T, disLoss, demandA_B)    
       
    ########################################################################### repair all     
    for item in cross_set:
        if dic_strNode.get(item,0)==0: continue
        i = dic_strID[item]
        if df_structure[r][i]>df_structure['WD_C10'][i] or \
            (item not in curIvtSet and df_structure[r][i]>df_structure['WD_C5'][i]):
                
            updateGraph(G, stru_damage, link_damage, item, True)
            computeRepairCost(r, crossingRC, repairC3, 'WD_C5', item)

    computeNewGraphCost(dic_graph, 'Crossing', G, baseline, iso, crossingT, isoTrip_sum, num, T, disLoss, demandA_B)

    reBridge_s = np.sum(repairC1)/1e6  
    reCulvert_s = np.sum(repairC2)/1e6  
    reCrossing_s = np.sum(repairC3)/1e6  
    repairCost =  reBridge_s+ reCulvert_s + reCrossing_s                  
    
    return disLoss, isoTrip_sum,link_damage,stru_damage,repairCost,dic_graph

# In[]: water flood user disruption cost 
# expected annual user 

def allDisrupt(curIvtSet, baseline):
    
    disUC=[]    
    for i in range(10):
        start = time.clock()    
        disUC.append(disrupt(rperiod[i],G2,curIvtSet, baseline,1,1,1,1,1,1,T,1)) # total $ value of extra user cost because of disruption
        print(time.clock() - start, 'seconds')  
    EAUC=0
    EAUL=0  
    for i in range(9):                
        EAUC = EAUC + (1.0/rpTime[i]-1.0/rpTime[i+1])*(np.sum(disUC[i][0])+np.sum(disUC[i+1][0]))
        EAUL = EAUL + (1.0/rpTime[i]-1.0/rpTime[i+1])* (np.sum(disUC[i][1])+np.sum(disUC[i+1][1]))
    EAUC = EAUC/2
    EAUL = EAUL/2    
    print('Expected user disruption cost is $', EAUC,'million per year', '\n',
          'Expected isolation trips is',EAUL,'million per year.')
    
    return disUC, EAUC, EAUL
        
# In[] construct the investment set. 
# if invest, we invest all structures with the same type within a link if the structure damaged criteria 
# are the same, in this case, if they have water depth, we invest.

idx=0
dic_idxGroup={} # structure set ID to structure ID. we assign all structures in a link as a set if it has WD
dic_g2Link={}
dic_g2Type={}
num_str=0

for link, strList in dic_linkStr.items():
    bridge_set=set()
    culvert_set=set()
    cross_set=set()
    for item in strList:
        if dic_strType[item]=='Bridge' and df_structure['WD_C100'][dic_strID[item]]>0: 
            bridge_set.add(item)
            num_str+=1
        elif dic_strType[item]=='Culvert'and df_structure['WD_C50'][dic_strID[item]]>0:
            culvert_set.add(item)
            num_str+=1
        elif dic_strType[item]=='Crossing'and df_structure['WD_C10'][dic_strID[item]]>0:
            cross_set.add(item)
            num_str+=1
            
    if bridge_set:       
        dic_idxGroup[idx]=bridge_set
        dic_g2Link[idx]=link
        dic_g2Type[idx]='Bridge'
        idx+=1
    if culvert_set:
        dic_idxGroup[idx]=culvert_set
        dic_g2Link[idx]=link
        dic_g2Type[idx]='Culvert'
        idx+=1
    if cross_set:
        dic_idxGroup[idx]=cross_set
        dic_g2Link[idx]=link
        dic_g2Type[idx]='Crossing'
        idx+=1
        
num_set=idx        
dic_linkNode={}
dic_linkRowID={}
for i in range(len(gdf2)):
    dic_linkNode[gdf2['OBJECT_ID'][i]]=(gdf2['FNODE_'][i], gdf2['TNODE_'][i])
    dic_linkRowID[gdf2['OBJECT_ID'][i]]=i
                 
                 
                 
# In[]
def computeDisruptCost(graph, link, node, baseline, days, cost, iso): # benefit of adding a link back to the graph
    ivtBnf = 0
    isoBnf = 0
    
    temp = graph[node[0]][node[1]]['total_cost'] # this is the disaster graph
    # if the weight < 1e9, the link is not broken at all under this disaster. Thus the investment has no benefit.
    if temp<1e9:
        return ivtBnf, isoBnf
    
    graph[node[0]][node[1]]['total_cost'] = gdf2['total_cost'][dic_linkRowID[link]] # change the travel cost to original graph
    to_allNode = []  
    for j in range(len(od)): 
        to_allNode.append(nx.single_source_dijkstra_path_length(graph,od[j],weight = 'total_cost'))  
    graph[node[0]][node[1]]['total_cost'] = temp # change the investment graph back to the disaster graph

    cost_disrupt= np.zeros((len(od),len(od)))   
    for j in range(len(od)): # for each OD pair 
        for k in range(j+1,len(od)):
            weight = to_allNode[j].get(od[k])
            if weight >= 1e9:
                isoBnf += T[j][k] * days / 1e6
                cost_disrupt[j][k] = compute_usrCost(baseline[j][k], T[j][k], 20*baseline[j][k])
            else:
                cost_disrupt[j][k] = compute_usrCost(baseline[j][k], T[j][k], weight)
            
    ivtBnf = np.sum(cost_disrupt) * days / 1e6
                   
    ivtBnf = cost - ivtBnf # system original cost - investment benefit (million dollar per year)
    isoBnf = iso - isoBnf # system original isolation trips - isolation trips after invest the link
    
    return ivtBnf, isoBnf
# In[]
def compute_benef(baseline, disUC, idx, n):
    # input: 
        # baseline: the user cost between each od pair without any disaster
        # penalty: penalty for od disconnect
        # disUC: object saving network under each period, return values of function "disrupt"
        # idx: structure group id
        # n: period id
    # output:
        # costBnf: benefit of investing idx
        # isoTBnf: benefit of avoiding od disconnection by investing idx
        
    link = dic_g2Link[idx]
    groupType = dic_g2Type[idx]
    node = dic_linkNode[link]
    dic_graph = disUC[n][5]
    stru_damage = disUC[n][3]
    
    costBnf = 0
    isoTBnf = 0
    
    # The first case: the last 150 days after a disaster. All culverts and crossings are recovered. Only
    # Bridges are in repair. The corresponding network is dic_graph['Bridge']. There is no benefit for
    # culvert and crossing since they've already been recovered anyway. Only bridges are considered.
    # Call computeDisrupt to recover the invested bridge group and compute its benefit for network.
    if groupType == 'Bridge':
        benefit, isotrips = computeDisruptCost(dic_graph[0], \
                                                     link, node, baseline, 150, disUC[n][0][0], disUC[n][1][0])
        costBnf += benefit
        isoTBnf += isotrips       
    # The second case: 24 days in the middle, where in the network crossings are recovered. The corresponding
    # network is dic_graph['Culvert']. Crossing investment has no benefit in this period. If the structure is
    # culvert, since it is not destroyed in this return period, the bridges on the same link must not been
    # destroyed. Thus the link is guaranteed to be not broken. If the structure is bridge, the culverts on this
    # link may be destroyed.
    if groupType == 'Culvert' or groupType == 'Bridge':
        roadDestroy = False # link destroy or not
        # if invest structure is bridge, justify if the link is broken
        if groupType == 'Bridge':
            for item in dic_linkStr[link]:
                # if any culvert on this link is in stru_damage set, the link is broken.
                if dic_strType[item] == 'Culvert' and item in stru_damage:
                    roadDestroy = True
                    break
        # compute benefit only if link is not broken
        if not roadDestroy:
            benefit, isotrips = computeDisruptCost(dic_graph[1], \
                                                         link, node, baseline, 24, disUC[n][0][1], disUC[n][1][1])
            costBnf += benefit
            isoTBnf += isotrips
    
    # The third case: the first 7 days, where all structures are potentially destroyed. The network is
    # dic_graph['Crossing']. For invest bridge, need to make sure culverts and crossings on this link
    # are not broken. For invest culvert, bridge must be in good condition, check crossings. For invest
    # crossing, since it is not broken, bridges and culverts must be intact.
    roadDestroy = False
    for item in dic_linkStr[link]:
        # check culverts and crossings for bridge
        if groupType == 'Bridge' and dic_strType[item] != 'Bridge' \
                                                and dic_strType[item] != 'Footbridge' and item in stru_damage:
            roadDestroy = True
            break
        # check crossings for culvert
        elif groupType == 'Culvert' and dic_strType[item] == 'Crossing' and item in stru_damage:
            roadDestroy = True
            break
    # invest crossing does not need any checking. If link is not broken, compute benefit
    if not roadDestroy:
        benefit, isotrips = computeDisruptCost(dic_graph[2], \
                                                     link, node, baseline, 7, disUC[n][0][2], disUC[n][1][2])
        costBnf += benefit
        isoTBnf += isotrips
    
    return costBnf, isoTBnf

# In[]
def investbenef (idx,rp,n,T,disUC,baseline):    
    # input: 
    # idx: index of groupset ID; 
    # rp: return period; 
    # n: rank of return period; 
    # dic_graph: disaster graph from last iteration
    # T: traffic flow
    # disUC
    
    # output:
    # costBnf, isoTBnf, repairBnf, ivtCost    
    
    bridgeRC = 40000*1e-6 # repair cost $
    culvertRC = 10000*1e-6
    crossingRC = 1000*1e-6
    
    bridgeIC = 4000*1e-6 # repair cost $
    culvertIC = 1000*1e-6
    crossingIC = 500*1e-6
    
    link = dic_g2Link[idx] # 
    groupSet = dic_idxGroup[idx] # structure ID
    groupType = dic_g2Type[idx]
    
    costBnf = 0
    isoTBnf = 0
    repairBnf=0
    ivtCost=0
    
    if groupType == 'Bridge':
        for item in groupSet:
            rowID = dic_strID[item]
            l = float(df_structure['Over_Lengt'][rowID]) 
            b = float(df_structure['Clear_Widt'][rowID])
            if df_structure[rp][rowID]<=df_structure['WD_C100'][rowID]:
                if item not in disUC[n][3]: continue
                repairBnf = repairBnf + (df_structure[rp][rowID] - df_structure['WD_C50'][rowID]) \
                            /(df_structure['WD_C1000'][rowID] \
                            - df_structure['WD_C50'][rowID]) * bridgeRC* l * b
                ivtCost = ivtCost + bridgeIC*l*b                  
            else:
                return costBnf, isoTBnf, repairBnf,ivtCost
                
      
        costBnf, isoTBnf = compute_benef(baseline, disUC, idx, n)
                        
        return costBnf, isoTBnf, repairBnf, ivtCost
    
    elif groupType == 'Culvert':
        for item in groupSet:
            rowID = dic_strID[item]
            l = float(df_structure['Over_Lengt'][rowID]) 
            b = float(df_structure['Clear_Widt'][rowID] )
            if df_structure[rp][rowID]<=df_structure['WD_C50'][rowID]:
                if item not in disUC[n][3]: continue
                repairBnf = repairBnf + (df_structure[rp][rowID] - df_structure['WD_C20'][rowID])/(df_structure['WD_C1000'][rowID] \
                               - df_structure['WD_C20'][rowID])* culvertRC* l * b
                ivtCost += culvertIC*l*b                
            else:
                return costBnf, isoTBnf, repairBnf,ivtCost
        
        for item in dic_linkStr[link]:
            if (dic_strType[item] == 'Bridge' or dic_strType[item] == 'Footbridge') \
               and item in disUC[n][3]:
                   return costBnf, isoTBnf, repairBnf,ivtCost
                
        costBnf, isoTBnf = compute_benef(baseline, disUC, idx, n)
                        
        return costBnf, isoTBnf, repairBnf, ivtCost
    
    else:
        for item in groupSet:
            rowID = dic_strID[item]
            l = float(df_structure['Over_Lengt'][rowID]) 
            b = float(df_structure['Clear_Widt'][rowID] )
            if (df_structure[rp][rowID] <= df_structure['WD_C10'][rowID]):
                if item not in disUC[n][3]: continue
                if np.isnan(l):
                    l=10                
                if np.isnan(b):
                    b=3                           
                repairBnf = repairBnf + (df_structure[rp][rowID] - df_structure['WD_C5'][rowID])/(df_structure['WD_C1000'][rowID] \
                             - df_structure['WD_C5'][rowID])*crossingRC* l * b                          
                ivtCost += crossingIC*l*b 
            else:
                return costBnf, isoTBnf, repairBnf,ivtCost

        for item in dic_linkStr[link]:
            if dic_strType[item] != 'Crossing' and item in disUC[n][3]:
                   return costBnf, isoTBnf, repairBnf,ivtCost
        
        costBnf, isoTBnf = compute_benef(baseline, disUC, idx, n)
                        
        return costBnf, isoTBnf, repairBnf, ivtCost
    
# In[] Initial condition
dic_benef = {}
curIvtSet = set()
# In[] 

dic_benef = {}
rperiod = ['WD_C5','WD_C10','WD_C20','WD_C50','WD_C75','WD_C100','WD_C200','WD_C250','WD_C500','WD_C1000']   
rpTime = [5, 10, 20, 50, 75, 100, 200, 250, 500, 1000] # return period (year)

for i in range(10):
    df_structure[rperiod[i]] = np.nan_to_num(df_structure[rperiod[i]])

disUC, EAUC, EAUL = allDisrupt(curIvtSet, baseCost)

start = time.clock()   
count=0
ibResult=[]
for idx, strSet in dic_idxGroup.items():
    C = 0 # benefit in cost disruption
    L = 0 # benefit in trip isolation
    ibResult = []
    if strSet.issubset(curIvtSet) or (dic_g2Link[idx] not in dic_linkRowID): continue
    count+=1
    for i in range(6):
        ibResult.append(investbenef(idx,rperiod[i],i,T,disUC,baseCost))  
        print(count, strSet, rperiod[i], '\n')                    
    for i in range(5):                
        C = C + (1.0/rpTime[i]-1.0/rpTime[i+1])*(ibResult[i][0]+ibResult[i+1][0])
        L = L + (1.0/rpTime[i]-1.0/rpTime[i+1])*(ibResult[i][1]+ibResult[i+1][1])
        
    C = C/2
    L = L/2     
        
    if C != 0:
        dic_benef[idx] = C # based on investGroup
        curIvtSet |= strSet
print(time.clock() - start, 'seconds')  
        
# In[]        
# save the investment structures and set relation  
curIvt ={} # investment set and structure      
for item in dic_benef.keys():
    curIvt[item]= list( dic_idxGroup[item])
              
# In[]
invList = list(curIvtSet)
invType={}
for item in invList:
    invType[item] = dic_strType[item]

# In[]
def compute_ivtCost(idx): # idx is the index of groupID
    
    bridgeIC = 4000*1e-6 # repair cost $
    culvertIC = 1000*1e-6
    crossingIC = 500*1e-6
    
    groupSet = dic_idxGroup[idx] # use idx of groupID find the structure ID
    groupType = dic_g2Type[idx]
    
    ivtCost = 0
    for item in groupSet:
        rowID = dic_strID[item]
        l = float(df_structure['Over_Lengt'][rowID]) 
        b = float(df_structure['Clear_Widt'][rowID] )
        if groupType == 'Bridge' or groupType == 'FootBridge':
            if np.isnan(l) or l==0:
                l=20 
            if np.isnan(b) or b==0:
                b=5     
            ivtCost += bridgeIC*l*b 
        elif groupType == 'Culvert':
            if np.isnan(l) or l==0:
                l=15 
            if np.isnan(b) or b==0:
                b=4     
            ivtCost += culvertIC*l*b 
        else:
            if np.isnan(l)or l==0:
                l=10 
            if np.isnan(b)or b==0:
                b=3             
            ivtCost += crossingIC*l*b 
    return ivtCost


# In[] # calculate the cost of investment a set
dic_ivtCost={}
for idx in dic_g2Link:
    dic_ivtCost[idx] = compute_ivtCost(idx)
# In[]
def opt_oneWeight(budget):   
        
    m=Model()
    x=m.addVars(len(dic_g2Link), vtype = GRB.BINARY, name = 'x')
    #m.setObjective(x.prod(dic_benef)+x.prod(dic_numStr)+x.prod(dic_crtStr), GRB.MAXIMIZE)
    m.setObjective(x.prod(dic_benef), GRB.MAXIMIZE)
    m.addConstr(x.prod(dic_ivtCost)<=budget)
    m.params.mipgap = 0.01
    
    m.optimize()
    
    soln= np.zeros(len(dic_g2Link)) # soln save the groupID
    n=0
    for v in m.getVars(): 
        print('%s %g' % (v.varName, v.x)) 
        soln[n] = int(v.x) 
        n+=1
    
    ivt_str=[] # structure ID of investment set
    univt_str=[]
    ivt_group =[] # structure group ID of investment set
    for i in range(len(soln)):
        if soln[i]==1:
            ivt_group.append(i)
            ivt_str.extend(dic_idxGroup[i])
        else:     
            univt_str.extend(dic_idxGroup[i])
    ivt_Cost = []
    for item in ivt_group: # find benefit from ivt_group
        ivt_Cost.append(dic_ivtCost[item])
        
    cost = np.sum(ivt_Cost)       
    return  cost, ivt_str, ivt_group, ivt_Cost   
# In[] this is used to run the optimal investment under budget with function opt_oneWeight
result=[]
ivt_str=[]
ivtCost=[]
bc=np.zeros((20,2))
for i in range(len(bc)):
    result.append(opt_oneWeight(i*5+60)) # this is the input of budget
    ivt_str.append(result[i][1]) # structures under budgets
    ivtCost.append(result[i][0]) # investment cost of structure

                 
# In[]  this calcualte the real investment benefits using the infratructure found in the optimization algorithm  
realValue=[]
realBenefit=[]  
disrupC =[]  
for i in range(len(bc)):    
    realValue.append(allDisrupt(set(ivt_str[i]), baseCost) ) 
    disrupC.append(realValue[i][1]) 
    realBenefit.append(141.12 - realValue[i][1])    

## In[] find the inventory for the benefit for each groupset
#ivt_benefit = []  
#ivt_numStr = []
#ivt_crtStr = [] 
#ivt_Cost = []
#for i in range(len(dic_g2Link)):
#    ivt_benefit.append(dic_benef.get(i,0))
#    ivt_numStr.append(dic_numStr.get(i,0))
#    ivt_crtStr.append(dic_crtStr.get(i,0))
#    ivt_Cost.append(dic_ivtCost.get(i,0))

# In[] an alternative model with three weight in objective function; comment the code below if you don't need it 
dic_benef_weighted={}
sumWeight1 = np.sum(list(dic_benef.values()))
for key in dic_benef:
    dic_benef_weighted[key] = float(dic_benef[key])/sumWeight1
# In[]
# weight from shortest path
weight_sp=[]
dic_numLink={} # record how many time the links has been used as shortest path
idx=0
for i in range(len(od)):
    for j in range(i+1, len(od)):
        for link in stpID[idx]:
            if link in dic_numLink:
                dic_numLink[link] += int(T[i][j])
            else:
                dic_numLink[link] = int(T[i][j])
        idx+=1

dic_numStr={} # record structure group to number of time that group has been used as shortest path
for idx, link in dic_g2Link.items():
    if link in dic_numLink:
        dic_numStr[idx]=dic_numLink[link]
    else:
        dic_numStr[idx]=0
                 
dic_numStr_weighted = {}                       
sumWeight2 = np.sum(list(dic_numStr.values()))
for idx in dic_numStr:
    dic_numStr_weighted[idx] = float(dic_numStr[idx])/ sumWeight2
# In[]               
# weight from criticality
dic_crtLink={}
data=[]
with open('criticality_mz.csv', 'r') as csvfile:   
    fileReader =  csv.reader(csvfile, delimiter=',')
    next(fileReader,None)
    fileReader=list(fileReader)
    for row in fileReader:
        dic_crtLink[float(row[0])]=float(row[1])

dic_crtStr={}
for idx, link in dic_g2Link.items():
    if link in dic_crtLink:
        dic_crtStr[idx]=dic_crtLink[link]
    else:
        dic_crtStr[idx]=0
                  
dic_crtStr_weighted={}  
               
sumWeight3 = np.sum(list(dic_crtStr.values()))
for idx in dic_crtStr:
    dic_crtStr_weighted[idx] = dic_crtStr[idx] / sumWeight3
# In[] optimization with disaster benefit weight, and criticality, vulnearability weight

def opt_TriWeight(budget):
#    dic_ivtCost={}
#    for idx in dic_g2Link:
#        dic_ivtCost[idx] = compute_ivtCost(idx)
        
    m=Model()
    x=m.addVars(len(dic_g2Link), vtype = GRB.BINARY, name = 'x')
    m.setObjective(x.prod(dic_benef_weighted)+x.prod(dic_numStr_weighted)+x.prod(dic_crtStr_weighted), GRB.MAXIMIZE)
    m.addConstr(x.prod(dic_ivtCost)<=budget)
    m.params.mipgap = 0.01
    
    m.optimize()
    
    soln= np.zeros(len(dic_g2Link))
    n=0
    for v in m.getVars(): 
        print('%s %g' % (v.varName, v.x)) 
        soln[n] = int(v.x) 
        n+=1
    
    ivt_str=[] # structure ID of investment set
    univt_str=[]
    ivt_group =[] # structure group ID of investment set
    for i in range(len(soln)):
        if soln[i]==1:
            ivt_group.append(i)
            ivt_str.extend(dic_idxGroup[i])
        else:     
            univt_str.extend(dic_idxGroup[i])
    
    ivt_benefit = []  
    ivt_Cost = []
    ivt_numStr = []
    ivt_crtStr = []
    
    for item in ivt_group:
        ivt_benefit.append(dic_benef_weighted.get(item,0))
        ivt_numStr.append(dic_numStr_weighted.get(item,0))
        ivt_crtStr.append(dic_crtStr_weighted.get(item,0))

        ivt_Cost.append(dic_ivtCost[item])
        
    obj= np.sum(ivt_benefit) + np.sum(ivt_numStr) + np.sum(ivt_crtStr)
    cost = np.sum(ivt_Cost)
    return  obj, cost, ivt_str, ivt_benefit, ivt_Cost                

# In[] this is used to run the optimal investment under budget with function opt_threeWeight
result1=[]
ivt_str1=[]
ivtCost1=[]
bc1=np.zeros((20,2))
for i in range(len(bc1)):
    result1.append(opt_TriWeight(i*5+60))
    ivt_str1.append(result1[i][2]) # structures under budgets
    ivtCost1.append(result1[i][1]) # investment cost of structure
    bc1[i,0]=result1[i][0]
    bc1[i,1]=result1[i][1]

# In[]    
realValue1=[]
realBenefit1=[]  
disrupC1 =[]  
for i in range(len(bc1)):    
    realValue1.append(allDisrupt(set(ivt_str1[i]), baseCost) ) 
    disrupC1.append(realValue1[i][1]) 
    realBenefit1.append(141.12 - realValue1[i][1])    

#with open("output.csv", "wb") as f:
#    writer = csv.writer(f)
#    for i in soln:
#        writer.writerow([i])              

