#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:42:50 2017

@author: yan
"""

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

# this file is used for vulnerability analysis for before and after investment. 
# before investment, ivtSet= {}
# after investment, ivtSet comes from optimizationV_10 file, change ivtSet to the corresponding island and disaster type 
# output: summary, and summary_damage
# In[]: Input flood and infrastructure data    
ivtSet = {}    
# mozambique input
network = r'./input/MZ_inputs/Road_all_floods.shp'
centroid = r'./input/MZ_inputs/OD_all_MZ_v1.shp'
dbf = Dbf5(r'./input/MZ_inputs/Bridge_all_floods_v1.dbf')
df_structure = dbf.to_dataframe()

# check if the water depth on df_structure is correct: it should be non-decrease from 5 return to 1000 return period
rperiod =['WD_C5','WD_C10','WD_C20','WD_C50','WD_C75','WD_C100','WD_C200','WD_C250','WD_C500','WD_C1000']   
rpTime = [5, 10, 20, 50, 75, 100, 200, 250, 500, 1000] # return period (year)

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
gdf_clean=gdf.iloc[:, 105:114]
gdf_clean=pd.concat([gdf_clean, gdf.loc[:,'OBJECTID']], axis=1)
gdf_clean = gdf_clean.rename(columns={'OBJECTID': 'OBJECT_ID'}) # rename to keep name consistent with Fiji data
          
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

# In[]: 
# baseline: find the shortest path for each od to minimize the total travel cost; 
# output: 1) baseCost ($): total travel cost between all OD pairs; 2) basePath : the shortest path between all OD pairs   

basePath = [[[]for i in range(len(od))] for j in range(len(od))]  # shortest path represented by node id
baseCost=np.zeros((len(od),len(od)))

for i in range(len(od)):      
    for j in range(i+1,len(od)):       
        basePath[i][j]=nx.dijkstra_path(G2,od[i],od[j],weight = 'total_cost') # shortest path represent by node ID
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
# In[]: categorize the type of structure           
df_structure['StructureT'] = ''
for i in range(len(df_structure)):
    print i
    if "Bridge" in str(df_structure['Str_Desc'][i]) or "bridge" in str(df_structure['Str_Desc'][i]):
        df_structure['StructureT'][i] = 'Bridge'                 
    elif "Culvert" in str(df_structure['Str_Desc'][i]):
        df_structure['StructureT'][i] = 'Culvert'                                         
    else:
        df_structure['StructureT'][i] = 'Crossing'                                         
                                          
# In[]:                                             
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

# In[]: economic loss using trapz rule to approximate the integral to accelarate computation speed, 
# because of large number of OD pair

# for each OD pair:
    
def deltaV(C_0, N_0, C_1,beta=2.9):
    
    if C_0==0:
        return 0,0,0
    
    N_1 = N_0*np.exp(-beta*np.log(C_1/C_0)) # demand after the travel cost increase (disruption cost)
    
    x = np.linspace(N_1, N_0, num=11)
    C = C_0*np.exp(-1/beta*np.log(x/N_0))-C_0
    surplus_loss = np.trapz(C, dx=(N_0-N_1)/10)+N_1*(C_1-C_0)

    return surplus_loss, N_1, N_0

# In[]:
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
#        start = time.clock()          
        to_allNode.append(nx.single_source_dijkstra_path_length(G,od[j],weight = 'total_cost'))  
#        print(time.clock() - start, 'seconds')  

    cost_disrupt= np.zeros((len(od),len(od)))                 
    for j in range(len(to_allNode)):
        for k in range(len(od)):
            if k>j:
               cost_disrupt[j][k] = to_allNode[j].get(od[k])

    for index, item in np.ndenumerate(cost_disrupt):
        if item>=1e10:
            cost_disrupt[index]=baseline[index]
            iso[index] += T[index] * days / 1e6               
            isoTrip_sum[loc] += T[index] * days / 1e6   
                                   
    num[loc] =  np.sum(np.multiply(cost_disrupt>baseline, T)) * days / 1e6   # number of disrupted trips at the duration (million)
    
    ecoLoss = np.zeros((len(od),len(od)))
    demandAfter = np.zeros((len(od),len(od)))
    demandBefore = np.zeros((len(od),len(od))) 
    n=0
    
   
    for j in range(len(od)): # for each OD pair 
        for k in range(j+1,len(od)):
            C_0 = baseCost[j][k]
            N_0 = T[j][k]
            C_1 = cost_disrupt[j][k]
            if N_0 !=0:
                ecoLoss[j][k],demandAfter[j][k],demandBefore[j][k] = deltaV(C_0, N_0, C_1)
            n+=1    
      
    disLoss[loc] = np.sum(ecoLoss)* days / 1e6 # disruption loss at that duration, we have 3 duration
    demandA_B[loc] = np.sum(demandBefore-demandAfter)*days/1e6 # demand changed
    
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
    
    bridgeT= 330*bridge_r # repair duration
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
    isoTrip_sum=[0,0,0]
    num = [0, 0, 0]
    disLoss=[0, 0, 0]
    demandA_B=[0,0,0]
    
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
            updateGraph(G, stru_damage, link_damage, item, False)
            computeRepairCost(r, bridgeRC, repairC1, 'WD_C100', item)
            
        elif item not in curIvtSet:
        
            if df_structure[r][i]>df_structure['WD_C50'][i]:                           
                updateGraph(G, stru_damage, link_damage, item, False)
                computeRepairCost(r, bridgeRC, repairC1, 'WD_C50', item)
#    start = time.clock() 
    computeNewGraphCost(dic_graph, 'Bridge', G, baseline, iso, bridgeT, isoTrip_sum, num, T, disLoss, demandA_B)
#    print(time.clock() - start, 'seconds')     

############################################################## repair culvert and building bailey bridge
    for item in bridge_set:
        if dic_strNode.get(item,0)==0: continue
        i = dic_strID[item]
        
        if item in curIvtSet and df_structure[r][i]>df_structure['WD_C100'][i]: # invest but disrupt, assume after invest, design standard=100 yr
            updateGraph(G, stru_damage, link_damage, item, True)
            repairC1.append(1e6)
            
        elif item not in curIvtSet:
        
            if df_structure[r][i]>df_structure['WD_C50'][i]:
                updateGraph(G, stru_damage, link_damage, item, True)
                repairC1.append(1e6)

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
#    start = time.clock()     
    computeNewGraphCost(dic_graph, 'Culvert', G, baseline, iso, culvertT, isoTrip_sum, num, T, disLoss, demandA_B)
#    print(time.clock() - start, 'seconds')     
       
    ###################################################################################### repair all     
    for item in cross_set:
        if dic_strNode.get(item,0)==0: continue
        i = dic_strID[item]
        if df_structure[r][i]>df_structure['WD_C10'][i] or \
            (item not in curIvtSet and df_structure[r][i]>df_structure['WD_C5'][i]):
                
            updateGraph(G, stru_damage, link_damage, item, True)
            computeRepairCost(r, crossingRC, repairC3, 'WD_C5', item)
                      
#    start = time.clock()         
    computeNewGraphCost(dic_graph, 'Crossing', G, baseline, iso, crossingT, isoTrip_sum, num, T, disLoss, demandA_B)
#    print(time.clock() - start, 'seconds')     

    #################################### isolation economic cost

    
    boatCost = baseline*10
    ecoLoss = np.zeros((len(od),len(od)))
    demandAfter = np.zeros((len(od),len(od)))
    demandBefore = np.zeros((len(od),len(od))) 
    for j in range(len(od)): # for each OD pair 
        for k in range(j+1,len(od)):
            C_0 = baseCost[j][k]
            N_0 = iso[j][k]
            C_1 = boatCost[j][k]
            if N_0 !=0:
                ecoLoss[j][k],demandAfter[j][k],demandBefore[j][k] = deltaV(C_0, N_0, C_1)

    isoLoss = np.sum(ecoLoss)
    isodemandA = np.sum(demandAfter)
    isolatrip = np.sum(iso)    

    reBridge_s = np.sum(repairC1)/1e6  
    reCulvert_s = np.sum(repairC2)/1e6  
    reCrossing_s = np.sum(repairC3)/1e6      
    
    return disLoss, num, isoTrip_sum,link_damage,stru_damage,reBridge_s, \
            reCulvert_s,reCrossing_s, demandA_B,isoLoss,isodemandA,isolatrip             
            
    # output: 
    # 0, disLoss: change of surplus because of disruption  
    # 1, num_disrupt: number of disrupted trips 
    # 2, isoTrip_sum: number of isolated trips 
    # 3, link_damage: damaged link ID      
    # 4, stru_damage: damaged structure ID
    # 5, reBridge_s: total repair cost of bridges (million$)
    # 6, reCulvert_s: total repair cost of culvert (million$)
    # 7, rep_Crossing: total repair cost of crossing
    # 8, demandA_B: total number of travelers changed because of disruption
    # 9, isoLoss: change of surplus because of isolation
    # 10, isodemandA: number of remaining isolation travelers 
    # 11, isotrip: number of total isolation travelers, it equals to the sum of isoTrip_sum    
            
# In[]: summary
baseline = baseCost 
disUC=[]    
for i in range(10):
    start = time.clock()    
    disUC.append(disrupt(rperiod[i],G2,ivtSet, baseline,1,1,1,1,1,1,T,1))
    print(time.clock() - start, 'seconds')  

disruptC=[]
isoLoss = []

disruptT =[] 
isolationT =[]
lossTripD = []
lossTripI=[]
lossTrip=[]
remainI=[]
remainD=[]
repairC =[]
numStr = []
numLink=[]

for i in range(10):
    disruptC.append(np.sum(disUC[i][0]))
    disruptT.append(np.sum(disUC[i][1]))
    isolationT.append(np.sum(disUC[i][2]))  
    numLink.append(len(disUC[i][3]))
    numStr.append(len(disUC[i][4]))
    repairC.append((disUC[i][5]+disUC[i][6]+disUC[i][7]))     
    lossTripD.append(np.sum(disUC[i][8]))
    isoLoss.append(np.sum(disUC[i][9]))    
    lossTripI.append(np.sum(disUC[i][11])-np.sum(disUC[i][10]))
    remainI.append (isolationT[i]-lossTripI[i])
    remainD.append (disruptT[i]-lossTripD[i])
    lossTrip.append(lossTripD[i]+lossTripI[i])
    
# calculate the total consumer surplus, V, million$ per year
import scipy.integrate as integrate
v=0 # surplus per year (million$)
surplus_d=0 # surplus per day
beta = 2.9
C = lambda x: C_0*np.exp(-1/beta*np.log(x/N_0))    
for j in range(len(od)): # for each OD pair 
    for k in range(j+1,len(od)):
        C_0 = baseCost[j][k]
        N_0 = T[j][k]
        if N_0 !=0:
            surplus = integrate.quad(C, 0, N_0,epsabs=0.1, epsrel=0.1,limit=1)[0] - C_0 * N_0 # surplus per OD pair
            surplus_d+=surplus
                       
v=surplus_d*365/1e6 
V= [v]*10
   
# find the annual expectation
EAUC=0
EAUL=0
isoC=0
EAR=0 
     
for i in range(9):                
    EAUC = EAUC + (1.0/rpTime[i]-1.0/rpTime[i+1])*(disruptC[i]+disruptC[i+1])
    isoC = isoC + (1.0/rpTime[i]-1.0/rpTime[i+1])*(isoLoss[i]+isoLoss[i+1])
    EAUL = EAUL + (1.0/rpTime[i]-1.0/rpTime[i+1])*(lossTrip[i]+lossTrip[i+1])
    EAR = EAR + (1.0/rpTime[i]-1.0/rpTime[i+1])*(repairC[i]+repairC[i+1])

   
EALD = EAUC/2   
EALI = isoC/2
ecoC = EALD+EALI
EALT = EAUL/2 
EARC = EAR/2 
print 'Expected total user disruption cost is $', EALD, 'million per year'
print 'Expected isolation loss is $', EALI,'million per year'
print 'Expected sum cost is $', ecoC,'million per year'
print 'Expected total isolation trips is',EALT,'million per year.'
print 'Expected repair  cost is $', EARC,'million per year'
# In[]: Find the maximum repair cost for all infratructure by link when completely destroy

bridgeRC = 40000/1e6 # repair cost $
culvertRC = 10000/1e6
crossingRC = 1000/1e6
maxRC=[] 
dic_linkCost={}   
for i in range(len(df_structure)):
    l = float(df_structure['Over_Lengt'][i]) 
    if np.isnan(l):
        l=10
    b = float(df_structure['Clear_Widt'][i] )
    if np.isnan(b):
        b=3
        
    item = df_structure['OBJECTID'][i]
    linkID=dic_strLink[item]
    
    if dic_strType[item] == 'Bridge' or dic_strType[item] == 'Footbridge': 
        maxRC.append(bridgeRC* l* b )
        dic_linkCost[linkID]=dic_linkCost.get(linkID,0)+bridgeRC* l* b 
             
    if dic_strType[item] == 'Culvert': 
        maxRC.append(culvertRC* l* b )
        dic_linkCost[linkID]=dic_linkCost.get(linkID,0)+culvertRC* l* b 
    if dic_strType[item] == 'Crossing': 
        maxRC.append(crossingRC * l* b)
        dic_linkCost[linkID]=dic_linkCost.get(linkID,0)+crossingRC* l* b 
    
sum_RC = np.sum(maxRC)         
# In[]: summary 
# 1-dv/v : 12th column of summary
summary = np.column_stack((rpTime,isolationT,remainI,lossTripI,isoLoss,\
                           disruptT,remainD,lossTripD,disruptC,\
                           np.array(disruptC)+np.array(isoLoss),lossTrip,V,\
                                   1-(np.array(disruptC)+np.array(isoLoss))/V,\
                           numLink, np.array(numLink)/float(len(dic_linkStr)),numStr, \
                          np.array(numStr)/float(len(dic_strID)),repairC,np.array(repairC)/float(sum_RC)))      
summaryAAL=[EARC,EALD,EALI,ecoC,EARC/sum_RC,ecoC/v]
# In[]: summary
# damage structure under all return period   
#d_str_index=[]
#name = []
#d_returnP=[]
#d_damCost=[]
#for i in range(10):
#     for item in disUC[i][5].keys():
#         d_str_index.append(item)
#         name.append(disaster)
#         d_returnP.append(rpTime[i])
#         d_damCost.append(disUC[i][5][item])
#    
#summary_damage=pd.DataFrame({'a':d_str_index,'b':name,'c':d_returnP,'d':d_damCost})


# In[]: investment cost
investID=[]
structureN=[]
roadN=[]
investC=[]

bridgeIC = 44000*1e-6 # repair cost $
culvertIC = 11000*1e-6
crossingIC = 1100*1e-6


for item in ivtSet:
    ivtCost=0
    rowID = dic_strID[item]
    investID.append(item)
    structureN.append(df_structure['StructureN'][rowID])
    roadN.append(df_structure['RoadName'][rowID])
    groupType =  df_structure['StructureT'][rowID]
    
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

    investC.append(ivtCost) 

summary_invest=pd.DataFrame({'a':investID,'b':structureN,'c':roadN,'d':investC})

sum(investC)







       