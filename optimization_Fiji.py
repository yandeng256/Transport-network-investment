#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 16:02:33 2017

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
# In[]: Network Preparation
# flood north
#network = r'./input/FJ_inputs/fj_roads_north_FRA_v6.shp'
#centroid = r'./input/FJ_inputs/OD_north_9cities.shp'
#dbf = Dbf5(r'./input/FJ_inputs/fj_bridges_north_FRA_v6.dbf')


# flood south
network = r'./input/FJ_inputs/fj_roads_south_FRA_v6.shp'
centroid = r'./input/FJ_inputs/OD_south_10Cities.shp'
dbf = Dbf5(r'./input/FJ_inputs/fj_bridges_south_FRA_v6.dbf')


df_structure = dbf.to_dataframe()
gdf_points, gdf_node_pos, gdf = net_p.prepare_centroids_network(centroid, network)
          
# Create Networkx MultiGraph object from the GeoDataFrame
G = net_p.gdf_to_simplified_multidigraph(gdf_node_pos, gdf, simplify=False)

# Change the MultiGraph object to Graph object to reduce computation cost 
G_tograph = net_p.multigraph_to_graph(G)

# Observe the properties of the Graph object    
#print('number of disconnected compoents is', nx.number_connected_components(G_tograph))
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
gdf_points2['population'][6]=30000 # the population of Rakiraki is 29700 at 1996 census, 30000 as approximate
G2_multi = net_p.gdf_to_simplified_multidigraph(gdf_node_pos2, gdf_new, simplify=False)
G2 = net_p.multigraph_to_graph(G2_multi)
gdf2 = net_p.graph_to_df(G2)
allNode = G2.nodes()
allEdge = G2.edges()
od = gdf_points2['Node']

# the output of this section is gdf_points2, gdf_node_pos2, gdf2, G2

# In[]: baseline: find the shortest path for each od to minimize the total travel cost; 
# output: 1) baseCost ($): total travel cost between all OD pairs; 2) basePath : the shortest path between all OD pairs   

n=0
basePath = [[[]for i in range(len(od))] for j in range(len(od))]
baseCost=np.zeros((len(od),len(od)))
for i in range(len(od)):    
    for j in range(i+1,len(od)):
        print(n)
        basePath[i][j]=nx.dijkstra_path(G2,od[i],od[j],weight = 'total_cost')
        baseCost[i][j]=nx.dijkstra_path_length(G2,od[i],od[j],weight = 'total_cost')
        n=n+1

# baseline: find the shortest path for each od to minimize the total travel cost; 
# output: 1) baseLength: total travel distance between all OD pairs;
n=0    
baseLength=np.zeros((len(od),len(od)))
for i in range(len(od)):    
    for j in range(i+1,len(od)):
        print(n)
        baseLength[i][j]=nx.dijkstra_path_length(G2,od[i],od[j],weight = 'length')
        n=n+1
        
# In[]:  Dictionary 
stpID = [[]for i in range(45)]
dict_linkNode={}    
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
################### traffic flow matrix ####################################################
# OD table 
k=1.3
popu={}

for i in range(len(od)):
    idx = gdf_points2['Node'].tolist().index(od[i])
    popu[od[i]]=gdf_points2['population'].tolist()[idx]
    
T=np.zeros((len(od),len(od)))  # identical with OD order      
for i in range(len(od)):
    for j in range(i+1,len(od)):
        T[i][j]= k*popu[od[i]]*popu[od[j]]/baseLength[i][j]/30*1e3/50 # AADT (travel per day)

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
    dic_strLink[df_structure['OBJECTID'][i]] = df_structure['LINK_ID'][i]
    
    if df_structure['LINK_ID'][i] in gdf2['OBJECT_ID'].tolist():
        idx = gdf2['OBJECT_ID'][gdf2['OBJECT_ID'] == df_structure['LINK_ID'][i]].index[0]
        dic_linkID[df_structure['OBJECTID'][i]] = idx
        node = (gdf2['FNODE_'][idx],gdf2['TNODE_'][idx])
        dic_strNode[df_structure['OBJECTID'][i]]=node

# dictionary for link to all structure
dic_linkStr={} # link ID to ID of all structures in the link
for i in range(len(df_structure)):
    item = df_structure['LINK_ID'][i]
    if item in dic_linkStr:
        dic_linkStr[item].append(df_structure['OBJECTID'][i])#[dic_linkStr[item],df_structure['OBJECTID'][i]]
    else:
        dic_linkStr[item] = [df_structure['OBJECTID'][i]]
        
# In[]
def compute_usrCost(C_0, N_0, C_1, beta=2.9):
    
    C = lambda x: C_0*np.exp(-1/beta*np.log(x/N_0))    
    N_1 = N_0*np.exp(-beta*np.log(C_1/C_0))  
    
    area_1 = integrate.quad(C, 0, N_0)[0] - C_0 * N_0
    area_2 = integrate.quad(C, 0, N_1)[0] - C_1 * N_1                    
    return area_1 - area_2


# In[]: water flood user disruption cost
#def floodDisrupt(r):
# assumption: when a structure in the link is being disrupted, the link is disrupted   
# structure threshold: bridge 50, culvert 20, crossing 5 return period

def disrupt(r,graph,curIvtSet,baseline):
    # Input:
    #       r: return period
    #       graph: original traffic network
    #       curIvtSet: links already invested
    #       baseline: cost with no link disruption
    # Output:
    #       diff: difference of total user cost under flood r vs. baseline
    #       isoTrip_sum: total # of isolated trips under r
    #       link_damage: ID of damaged links
    #       stru_damage: ID of damaged structures
    #       repC_link: repair cost per link
     
    bridgeRC = 40000 # repair cost $
    culvertRC = 10000
    crossingRC = 1000
    
    stru_damage=set()    # ID of the bridge disrupted in that return period
    link_damage=[]    # OBJECT_ID of the road disrupted in that return period:
    repairC1 = [] # total repair cost in that return period (bridge)
    repairC2 = [] # (culvert)
    repairC3 = [] # crossing
    repC_link = {}
    dic_graph = {}
    
    bridge_set=set()
    culvert_set=set()
    cross_set=set()
    isoTrip_sum=[0,0,0]
    diff = [0, 0, 0]

    G = copy.deepcopy(graph)
    
    for link, strList in dic_linkStr.items():
        for item in strList:
            if dic_strType[item] == 'Bridge' or dic_strType[item] == 'Footbridge': 
                bridge_set.add(item)
            elif dic_strType[item] == 'Culvert':
                culvert_set.add(item)
            else:
                cross_set.add(item)
    
    for item in bridge_set:
        if dic_strNode.get(item,0)==0: continue
        i = dic_strID[item]
        if df_structure[r][i]>df_structure['WD_PU_100'][i] or \
            (item not in curIvtSet and df_structure[r][i]>df_structure['WD_PU_50'][i]):
            stru_damage.add(item)
            if not dic_strLink[item] in link_damage:
                link_damage.append(dic_strLink[item])
            
            G[dic_strNode[item][0]][dic_strNode[item][1]]['total_cost']=1e10
            cost = (df_structure[r][i] - df_structure['WD_PU_50'][i])/(df_structure['WD_PU_1000'][i] \
                       - df_structure['WD_PU_50'][i])*1*bridgeRC*float(df_structure['Length'][i])*\
                       float(df_structure['TotalWidth'][i])                            
            repairC1.append(cost) 
            repC_link[dic_strLink[item]] = repC_link.get(dic_strLink[item],0)+cost
    
    dic_graph['Bridge'] = copy.deepcopy(G)
    to_allNode = []   
    for j in range(len(od)): 
        to_allNode.append(nx.single_source_dijkstra_path_length(G,od[j],weight = 'total_cost'))       

    cost_disrupt= np.zeros((len(od),len(od)))                 
    for j in range(len(od)): # for each OD pair 
        for k in range(j+1,len(od)):
            weight = to_allNode[j].get(od[k])
            if weight >= 1e9:
                isoTrip_sum[0] += T[j][k] * 150 / 1e6
                cost_disrupt[j][k] = compute_usrCost(baseline[j][k], T[j][k], 10*baseline[j][k])
            else:
                cost_disrupt[j][k] = compute_usrCost(baseline[j][k], T[j][k], weight)
               
    diff[0] = np.sum(cost_disrupt) * 150 / 1e6
               
    for item in culvert_set:
        if dic_strNode.get(item,0)==0: continue
        i = dic_strID[item]
        if df_structure[r][i]>df_structure['WD_PU_50'][i] or \
            (item not in curIvtSet and df_structure[r][i]>df_structure['WD_PU_20'][i]):
            stru_damage.add(item)
            if not dic_strLink[item] in link_damage:
                link_damage.append(dic_strLink[item])
            
            G[dic_strNode[item][0]][dic_strNode[item][1]]['total_cost']=1e10
            cost = (df_structure[r][i] - df_structure['WD_PU_20'][i])/(df_structure['WD_PU_1000'][i] \
                       - df_structure['WD_PU_20'][i])*1*culvertRC*float(df_structure['Length'][i])*\
                       float(df_structure['TotalWidth'][i])                            
            repairC2.append(cost) 
            repC_link[dic_strLink[item]] = repC_link.get(dic_strLink[item],0)+cost
    
    dic_graph['Culvert'] = copy.deepcopy(G)        
    to_allNode = []
    for j in range(len(od)): 
        to_allNode.append(nx.single_source_dijkstra_path_length(G,od[j],weight = 'total_cost'))       

    cost_disrupt= np.zeros((len(od),len(od)))                 
    for j in range(len(od)): # for each OD pair 
        for k in range(j+1,len(od)):
            weight = to_allNode[j].get(od[k])
            if weight >= 1e9:
                isoTrip_sum[1] += T[j][k] * 24 / 1e6
                cost_disrupt[j][k] = compute_usrCost(baseline[j][k], T[j][k], 10*baseline[j][k])
            else:
                cost_disrupt[j][k] = compute_usrCost(baseline[j][k], T[j][k], weight)
            
    diff[1] = np.sum(cost_disrupt) * 24 / 1e6
               
    for item in cross_set:
        if dic_strNode.get(item,0)==0: continue
        i = dic_strID[item]
        if df_structure[r][i]>df_structure['WD_PU_10'][i] or \
            (item not in curIvtSet and df_structure[r][i]>df_structure['WD_PU_5'][i]):
            stru_damage.add(item)
            if not dic_strLink[item] in link_damage:
                link_damage.append(dic_strLink[item])
            
            G[dic_strNode[item][0]][dic_strNode[item][1]]['total_cost']=1e10
            l = float(df_structure['Length'][i]) 
            if np.isnan(l):
                l=10
            b = float(df_structure['TotalWidth'][i] )
            if np.isnan(b):
                b=3   
                
            cost = (df_structure[r][i] - df_structure['WD_PU_5'][i])/(df_structure['WD_PU_1000'][i] \
                       - df_structure['WD_PU_5'][i])*1*crossingRC*l*b                  
            repairC3.append(cost)                 
            repC_link[dic_strLink[item]] = repC_link.get(dic_strLink[item],0)+cost            
    
    dic_graph['Crossing'] = copy.deepcopy(G)                 
    to_allNode = []  
    for j in range(len(od)): 
        to_allNode.append(nx.single_source_dijkstra_path_length(G,od[j],weight = 'total_cost'))       

    cost_disrupt= np.zeros((len(od),len(od)))                 
    for j in range(len(od)): # for each OD pair 
        for k in range(j+1,len(od)):
            weight = to_allNode[j].get(od[k])
            if weight >= 1e9:
                isoTrip_sum[2] += T[j][k] * 7 / 1e6
                cost_disrupt[j][k] = compute_usrCost(baseline[j][k], T[j][k], 10*baseline[j][k])
            else:
                cost_disrupt[j][k] = compute_usrCost(baseline[j][k], T[j][k], weight)
            
    diff[2] = np.sum(cost_disrupt) * 7 / 1e6
           
    return diff, isoTrip_sum,link_damage,stru_damage,repC_link, dic_graph

# In[]: water flood user disruption cost 
# expected annual user 

def allDisrupt(curIvtSet, baseline):
#    rperiod = ['WD_PU_5','WD_PU_10','WD_PU_20','WD_PU_50','WD_PU_75','WD_PU_100',]   
#    rpTime = [5, 10, 20, 50, 75, 100] # return period (year)

    disUC=[]
    
    for i in range(10):
        start = time.clock()    
        disUC.append(disrupt(rperiod[i],G2,curIvtSet, baseline)) # total $ value of extra user cost because of disruption
        print(time.clock() - start, 'seconds')  

    ##uniq=set(disUC1[2])^set(disUC2[2])
    ##uimpa=uniq.intersection(link_s)
    EAUC=0
    EAUL=0  
    for i in range(9):                
        EAUC = EAUC + (1.0/rpTime[i]-1.0/rpTime[i+1])*(np.sum(disUC[i][0])+np.sum(disUC[i+1][0]))
        EAUL = EAUL + (1.0/rpTime[i]-1.0/rpTime[i+1])* (np.sum(disUC[i][1])+np.sum(disUC[i+1][1]))
    EAUC = EAUC/2
    EAUL = EAUL/2    
    print('Expected user disruption cost is $', EAUC,'million per year', '\n',
          'Expected isolation trips is',EAUL,'million per year.')
    
    # total repair cost 
#    EARC=0
#    for i in range(9):                
#        EARC += (1.0/rpTime[i]-1.0/rpTime[i+1])*(np.sum(disUC[i][4])+np.sum(disUC[i+1][4]))
#    EARC = EARC/2
#    print('Expected repair  cost is $', EARC,'million per year')
    
    return disUC, EAUC, EAUL
        
# In[] set and structure dictionary
idx=0
dic_idxGroup={} # setID to structure ID
dic_g2Link={}
dic_g2Type={}
for link, strList in dic_linkStr.items():
    bridge_set=set()
    culvert_set=set()
    cross_set=set()
    for item in strList:
        if dic_strType[item]=='Bridge' or dic_strType[item]=='Footbridge':
            bridge_set.add(item)
        elif dic_strType[item]=='Culvert':
            culvert_set.add(item)
        else:
            cross_set.add(item)
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
        
dic_linkNode={}
dic_linkRowID={}
for i in range(len(gdf2)):
    dic_linkNode[gdf2['OBJECT_ID'][i]]=(gdf2['FNODE_'][i], gdf2['TNODE_'][i])
    dic_linkRowID[gdf2['OBJECT_ID'][i]]=i
                 
# In[]
def computeDisruptCost(graph, link, node, baseline, days, cost, iso): # benefit of adding a link back to the graph
    ivtBnf = 0
    isoBnf = 0
    
    temp = graph[node[0]][node[1]]['total_cost']
    # if the weight < 1e9, the link is not broken at all under this disaster. Thus the investment has no benefit.
    if temp<1e9:
        return ivtBnf, isoBnf
    
    graph[node[0]][node[1]]['total_cost'] = gdf2['total_cost'][dic_linkRowID[link]]
    to_allNode = []  
    for j in range(len(od)): 
        to_allNode.append(nx.single_source_dijkstra_path_length(graph,od[j],weight = 'total_cost'))  
    graph[node[0]][node[1]]['total_cost'] = temp

    cost_disrupt= np.zeros((len(od),len(od)))   
    for j in range(len(od)): # for each OD pair 
        for k in range(j+1,len(od)):
            weight = to_allNode[j].get(od[k])
            if weight >= 1e9:
                isoBnf += T[j][k] * days / 1e6
                cost_disrupt[j][k] = compute_usrCost(baseline[j][k], T[j][k], 10*baseline[j][k])
            else:
                cost_disrupt[j][k] = compute_usrCost(baseline[j][k], T[j][k], weight)
            
    ivtBnf = np.sum(cost_disrupt) * days / 1e6
                   
    ivtBnf = cost - ivtBnf
    isoBnf = iso - isoBnf
    
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
        benefit, isotrips = computeDisruptCost(dic_graph['Bridge'], \
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
            benefit, isotrips = computeDisruptCost(dic_graph['Culvert'], \
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
        benefit, isotrips = computeDisruptCost(dic_graph['Crossing'], \
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
    
    bridgeIC = 44000*1e-6 # repair cost $
    culvertIC = 11000*1e-6
    crossingIC = 1100*1e-6
    
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
            l = float(df_structure['Length'][rowID]) 
            b = float(df_structure['TotalWidth'][rowID])
            if df_structure[rp][rowID]<=df_structure['WD_PU_100'][rowID]:
                if item not in disUC[n][3]: continue
                repairBnf = repairBnf + (df_structure[rp][rowID] - df_structure['WD_PU_50'][rowID]) \
                            /(df_structure['WD_PU_1000'][rowID] \
                            - df_structure['WD_PU_50'][rowID]) * bridgeRC* l * b
                ivtCost = ivtCost + bridgeIC*l*b                  
            else:
                return costBnf, isoTBnf, repairBnf,ivtCost
                
      
        costBnf, isoTBnf = compute_benef(baseline, disUC, idx, n)
                        
        return costBnf, isoTBnf, repairBnf, ivtCost
    
    elif groupType == 'Culvert':
        for item in groupSet:
            rowID = dic_strID[item]
            l = float(df_structure['Length'][rowID]) 
            b = float(df_structure['TotalWidth'][rowID] )
            if df_structure[rp][rowID]<=df_structure['WD_PU_50'][rowID]:
                if item not in disUC[n][3]: continue
                repairBnf = repairBnf + (df_structure[rp][rowID] - df_structure['WD_PU_20'][rowID])/(df_structure['WD_PU_1000'][rowID] \
                               - df_structure['WD_PU_20'][rowID])* culvertRC* l * b
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
            l = float(df_structure['Length'][rowID]) 
            b = float(df_structure['TotalWidth'][rowID] )
            if (df_structure[rp][rowID] <= df_structure['WD_PU_10'][rowID]):
                if item not in disUC[n][3]: continue
                if np.isnan(l):
                    l=10                
                if np.isnan(b):
                    b=3                           
                repairBnf = repairBnf + (df_structure[rp][rowID] - df_structure['WD_PU_5'][rowID])/(df_structure['WD_PU_1000'][rowID] \
                             - df_structure['WD_PU_5'][rowID])*crossingRC* l * b                          
                ivtCost += crossingIC*l*b 
            else:
                return costBnf, isoTBnf, repairBnf,ivtCost

        for item in dic_linkStr[link]:
            if dic_strType[item] != 'Crossing' and item in disUC[n][3]:
                   return costBnf, isoTBnf, repairBnf,ivtCost
        
        costBnf, isoTBnf = compute_benef(baseline, disUC, idx, n)
                        
        return costBnf, isoTBnf, repairBnf, ivtCost
    
# In[] 'WD_PU_10' rpTime = [5, 10, 20, 50, 75, 100, 200, 250, 500, 1000]

dic_benef = {}
dic_trip = {}
curIvtSet = set()

rperiod = ['WD_PU_5','WD_PU_10','WD_PU_20','WD_PU_50','WD_PU_75','WD_PU_100','WD_PU_200','WD_PU_250','WD_PU_500','WD_PU_1000']   
rpTime = [5, 10, 20, 50, 75, 100,200,250,500,1000] # return period (year)
for i in range(10):
    df_structure[rperiod[i]] = np.nan_to_num(df_structure[rperiod[i]])

disUC, EAUC, EAUL = allDisrupt(curIvtSet, baseCost)

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
        
# In[] return the type of the structures in curIvtSet
invList = list(curIvtSet)
invType={}
for item in invList:
    invType[item] = dic_strType[item]

# In[]
def compute_ivtCost(idx):
    
    bridgeIC = 44000*1e-6 # repair cost $
    culvertIC = 11000*1e-6
    crossingIC = 1100*1e-6
    
    groupSet = dic_idxGroup[idx]
    groupType = dic_g2Type[idx]
    
    ivtCost = 0
    for item in groupSet:
        rowID = dic_strID[item]
        l = float(df_structure['Length'][rowID]) 
        b = float(df_structure['TotalWidth'][rowID] )
        if groupType == 'Bridge' or groupType == 'FootBridge':
            if np.isnan(l):
                l=20 
            if np.isnan(b):
                b=5     
            ivtCost += bridgeIC*l*b 
        elif groupType == 'Culvert':
            if np.isnan(l):
                l=15 
            if np.isnan(b):
                b=4     
            ivtCost += culvertIC*l*b 
        else:
            if np.isnan(l):
                l=10 
            if np.isnan(b):
                b=3             
            ivtCost += crossingIC*l*b 
    return ivtCost

# In[] The optimization 

def opt_oneWeight(budget):   
    dic_ivtCost={}
    for idx in dic_g2Link:
        dic_ivtCost[idx] = compute_ivtCost(idx)
        
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
            
    ivt_benefit = []  
    ivt_Cost = []
    for item in ivt_group: # find benefit from ivt_group
        ivt_benefit.append(dic_benef[item])
        ivt_Cost.append(dic_ivtCost[item])
        
    obj= np.sum(ivt_benefit)
    cost = np.sum(ivt_Cost)       
    return  obj, cost, ivt_str, ivt_group,ivt_benefit, ivt_Cost   

# In[] this is used to run the optimal investment under budget with function opt_oneWeight
result=[]
ivt_str=[]
ivtCost=[]

for i in range(10):
    result.append(opt_oneWeight(i*20))
    ivt_str.append(result[i][2]) # structures under budgets
    ivtCost.append(result[i][1]) # investment cost of structure
    
## In[] find the inventory for the benefit for each groupset
#dic_ivtCost={}
#for idx in dic_g2Link:
#    dic_ivtCost[idx] = compute_ivtCost(idx)
#ivt_benefit = []  
#ivt_numStr = []
#ivt_crtStr = [] 
#ivt_Cost = []
#for i in range(len(dic_g2Link)):
#    ivt_benefit.append(dic_benef.get(i,0))
#    ivt_numStr.append(dic_numStr.get(i,0))
#    ivt_crtStr.append(dic_crtStr.get(i,0))
#    ivt_Cost.append(dic_ivtCost.get(i,0))

# In[]

# Alternative model;
# the following the another model with three weights in objective function. comment it out if you don't need it.

sumWeight1 = np.sum(list(dic_benef.values()))
for key in dic_benef:
    dic_benef[key] /= sumWeight1

# weight from shortest path
weight_sp=[]
dic_numLink={}
idx=0
for i in range(len(od)):
    for j in range(i+1, len(od)):
        for link in stpID[idx]:
            if link in dic_numLink:
                dic_numLink[link] += T[i][j]
            else:
                dic_numLink[link] = T[i][j]
        idx+=1

dic_numStr={}
for idx, link in dic_g2Link.items():
    if link in dic_numLink:
        dic_numStr[idx]=dic_numLink[link]
    else:
        dic_numStr[idx]=0
                  
dic_numStr_old = dic_numStr
                       
sumWeight2 = np.sum(list(dic_numStr.values()))
for idx in dic_numStr:
    dic_numStr[idx] /= sumWeight2
# In[]               
# weight from criticality
dic_crtLink={}
data=[]
with open('criticalityResult_v2.csv', 'r') as csvfile:   
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
                  
dic_crtStr_old = dic_crtStr   
               
sumWeight3 = np.sum(list(dic_crtStr.values()))
for idx in dic_crtStr:
    dic_crtStr[idx] /= sumWeight3
# In[] optimization with three weight in objective function: disaster benefit weight, and criticality, vulnearability weight
budget = 10
def opt_TriWeight(budget):
    dic_ivtCost={}
    for idx in dic_g2Link:
        dic_ivtCost[idx] = compute_ivtCost(idx)
        
    m=Model()
    x=m.addVars(len(dic_g2Link), vtype = GRB.BINARY, name = 'x')
    m.setObjective(x.prod(dic_benef)+x.prod(dic_numStr)+x.prod(dic_crtStr), GRB.MAXIMIZE)
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
        ivt_benefit.append(dic_benef_old.get(i,0))
        ivt_numStr.append(dic_numStr_old.get(i,0))
        ivt_crtStr.append(dic_crtStr_old.get(i,0))
        ivt_Cost.append(dic_ivtCost[item])
        
    obj= np.sum(ivt_benefit) + np.sum(ivt_numStr) + np.sum(ivt_crtStr)
    cost = np.sum(ivt_Cost)
    return  obj, cost, ivt_str, ivt_benefit, ivt_Cost                
##with open("output.csv", "wb") as f:
##    writer = csv.writer(f)
##    for i in soln:
##        writer.writerow([i])              
