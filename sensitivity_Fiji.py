#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 12:26:18 2018

@author: yan
"""

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
# In[]: investment data
#: Input data

investOption1=[]
south_1=set()
north_1=set()
with open('option1.csv','rU') as f:    
    reader = csv.reader(f)
    for row in reader:
        investOption1.append(row) # first row south, second row north
        
for item in investOption1[0]:
    south_1.add(int(item))

for item in investOption1[1][0:130]:
    north_1.add(int(item))

investOption2=[]
south_2=set()
north_2=set()
with open('option2.csv','rU') as f:    
    reader = csv.reader(f)
    for row in reader:
        investOption2.append(row) # first row south, second row north
        
for item in investOption2[0]:
    south_2.add(int(item))

for item in investOption2[1][0:7]:
    north_2.add(int(item))  
####
northClean=[]
southClean=[] 
north_3=set()  
south_3=set()

 
with open('north_clean.csv','rU') as f:    
    reader = csv.reader(f)
    for row in reader:
        northClean.append(row) # first row south, second row north
with open('south_clean.csv','rU') as f:    
    reader = csv.reader(f)
    for row in reader:
        southClean.append(row) # first row south, second row north
for item in northClean[0]:
    north_3.add(int(item))
for item in southClean[0]:
    south_3.add(int(item))   
    
#       
##flood_north= {119, 145,469, 679,1112,683,811,1127,1134,1135} # government
#flood_north ={102, 119, 145, 605, 657, 670, 676, 679, 680, 683, 723, 724, 732, 811, 1112, 1118, 1127, 1231} #opt

# flood south
#flood_south={192,204,205,1198,821,278,1005,416,418,420,422,1212,235,883} # government
#flood_south={192, 204, 205, 821, 934, 935, 1177, 1198, 1814, 2276}# from PU optimal
#flood_south={3, 4, 22, 46,216,217,296,329,444,448,458,479,768,769,835,1010,1093,1094,1200,1841,2278} #from PU optimal
ivtSet = south_3 # investment set after investment, change this one when you change input file
  
# In[]: Input flood and infrastructure data    
    
# flood north
#network = r'./input/inputs_v6/fj_roads_north_FRA_v6.shp'
#centroid = r'./input/fiji_north/OD_north_9cities.shp'
#dbf = Dbf5(r'./input/inputs_v6/fj_bridges_north_FRA_v6.dbf')

# flood south
network = r'./input/inputs_v6/fj_roads_south_FRA_v6.shp'
centroid = r'./input/fiji_south/OD_south_10Cities.shp'
dbf = Dbf5(r'./input/inputs_v6/fj_bridges_south_FRA_v6.dbf')


df_structure = dbf.to_dataframe()

# check if the water depth on df_structure is correct: non decreasing from 5 return to 1000 return
#rperiod = ['WD_PU_5','WD_PU_10','WD_PU_20','WD_PU_50','WD_PU_75','WD_PU_100','WD_PU_200','WD_PU_250','WD_PU_500','WD_PU_1000']   
rperiod =['DPU17_5','DPU17_10','DPU17_20','DPU17_50','DPU17_75','DPU17_100','DPU17_200','DPU17_250','DPU17_500','DPU17_1000']   

rpTime = [5, 10, 20, 50, 75, 100, 200, 250, 500, 1000] # return period (year)

wrongInfo=[]    
for i in range(len(df_structure)):
    for j in range(9):
        if df_structure[rperiod[j]][i] > df_structure[rperiod[j+1]][i] :
            wrongInfo.append(df_structure['OBJECTID'][i])
            
for item in wrongInfo:
    for j in range(10):
        df_structure[rperiod[j]][item]=0 

gdf_points, gdf_node_pos, gdf = net_p.prepare_centroids_network(centroid, network)
gdf_clean=gdf.iloc[:, 165:174]
gdf_clean=pd.concat([gdf_clean, gdf.loc[:,'OBJECT_ID']], axis=1)
          
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
gdf_points2['population'][6]=30000 # the population of Rakiraki is 29700 at 1996 census, 30000 as approximate
G2_multi = net_p.gdf_to_simplified_multidigraph(gdf_node_pos2, gdf_new, simplify=False)
G2 = net_p.multigraph_to_graph(G2_multi)
gdf2 = net_p.graph_to_df(G2)
allNode = G2.nodes()
allEdge = G2.edges()
od = gdf_points2['Node']

# the output of this section is gdf_points2, gdf_node_pos2, gdf2, G2

# In[]: 
# baseline: find the shortest path for each od to minimize the total travel cost; 
# output: 1) baseCost ($): total travel cost between all OD pairs; 2) basePath : the shortest path between all OD pairs   

basePath = [[[]for i in range(len(od))] for j in range(len(od))]  # shortest path represented by node id
baseCost=np.zeros((len(od),len(od)))
for i in range(len(od)):        
    for j in range(i+1,len(od)):
        basePath[i][j]=nx.dijkstra_path(G2,od[i],od[j],weight = 'total_cost')
        baseCost[i][j]=nx.dijkstra_path_length(G2,od[i],od[j],weight = 'total_cost')

# baseline: find the shortest path for each od to minimize the total travel cost; 
# output: 1) baseLength: total travel distance between all OD pairs;   
baseLength=np.zeros((len(od),len(od)))
for i in range(len(od)):  
    for j in range(i+1,len(od)):
#        start = time.clock()
        baseLength[i][j]=nx.dijkstra_path_length(G2,od[i],od[j],weight = 'length')
#        print(time.clock() - start, 'seconds')
        
# Dictionary 
stpID = [[]for i in range((len(od)*(len(od)-1)))] # shortest path represented by link id
dict_linkNode={}    # key: node id as tuple  value: link id
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
            

###### build a dictionary to match structure (key) and F,TNODE_ (value); structure type, and index in df_structure
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

# In[]: Transfer the isolation trip to economic loss

# for each OD pair:
import scipy.integrate as integrate  

def deltaV(C_0, N_0, C_1,beta=2.9):
    
    if C_0==0:
        return 0,0,0       

    C = lambda x: C_0*np.exp(-1/beta*np.log(x/N_0))    
    N_1 = N_0*np.exp(-beta*np.log(C_1/C_0))  
    loss = integrate.quad(C,N_1,N_0)[0] - C_0*(N_0-N_1)+(C_1-C_0)*N_1
    return loss,N_1,N_0

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
    l = float(df_structure['Length'][i]) 
    if np.isnan(l):
        l=10
    b = float(df_structure['TotalWidth'][i] )
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
    #dic_graph[stru_name] = copy.deepcopy(G)
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
            cost_disrupt[index]=baseline[index]
            iso[index] += T[index] * days / 1e6               
            isoTrip_sum[loc] += T[index] * days / 1e6   
                                   
    num[loc] =  np.sum(np.multiply(cost_disrupt>baseline, T)) * days / 1e6   # number of disrupted trips at the duration

    ecoLoss = np.zeros((len(od),len(od)))
    demandAfter = np.zeros((len(od),len(od)))
    demandBefore = np.zeros((len(od),len(od))) 

    for j in range(len(od)): # for each OD pair 
        for k in range(j+1,len(od)):
            C_0 = baseCost[j][k]
            N_0 = T[j][k]
            C_1 = cost_disrupt[j][k]
            if N_0 !=0:
                ecoLoss[j][k],demandAfter[j][k],demandBefore[j][k] = deltaV(C_0, N_0, C_1)

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
    bridgeRC = 13000*bridgeRC_r # repair cost $
    culvertRC = 2500*culvertRC_r
    crossingRC = 1250*crossingRC_r
    
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
        
        if item in curIvtSet and df_structure[r][i]>df_structure['DPU17_100'][i]: # invest but disrupt, assume after invest, design standard=100 yr
            updateGraph(G, stru_damage, link_damage, item, False)
            computeRepairCost(r, bridgeRC, repairC1, 'DPU17_100', item)
            
        elif item not in curIvtSet:
        
            if df_structure['Priority'][i] == 'Low':
                if df_structure[r][i]>df_structure['DPU17_50'][i]:                           
                    updateGraph(G, stru_damage, link_damage, item, False)
                    computeRepairCost(r, bridgeRC, repairC1, 'DPU17_50', item)
            elif df_structure['Priority'][i]=='Medium' or df_structure['Priority'][i]=='High':
                if df_structure[r][i]>df_structure['DPU17_20'][i]:
                    updateGraph(G, stru_damage, link_damage, item, False)
                    computeRepairCost(r, bridgeRC, repairC1, 'DPU17_20', item)
            else:
                if df_structure[r][i]>df_structure['DPU17_10'][i]:
                    updateGraph(G, stru_damage, link_damage, item, False)
                    computeRepairCost(r, bridgeRC, repairC1, 'DPU17_10', item)

    computeNewGraphCost(dic_graph, 'Bridge', G, baseline, iso, bridgeT, isoTrip_sum, num, T, disLoss, demandA_B)
    

############################################################## repair culvert and building bailey bridge
    for item in bridge_set:
        if dic_strNode.get(item,0)==0: continue
        i = dic_strID[item]
        
        if item in curIvtSet and df_structure[r][i]>df_structure['DPU17_100'][i]: # invest but disrupt, assume after invest, design standard=100 yr
            updateGraph(G, stru_damage, link_damage, item, True)
            repairC1.append(1e6)
            
        elif item not in curIvtSet:
        
            if df_structure['Priority'][i] == 'Low':
                if df_structure[r][i]>df_structure['DPU17_50'][i]:
                    updateGraph(G, stru_damage, link_damage, item, True)
                    repairC1.append(1e6)
            elif df_structure['Priority'][i]=='Medium' or df_structure['Priority'][i]=='High':
                if df_structure[r][i]>df_structure['DPU17_20'][i]:                           
                    updateGraph(G, stru_damage, link_damage, item, True)
                    repairC1.append(1e6)
            else:
                if df_structure[r][i]>df_structure['DPU17_10'][i]:
                    updateGraph(G, stru_damage, link_damage, item, True)
                    repairC1.append(1e6)  

    for item in culvert_set:
        if dic_strNode.get(item,0)==0: continue
        i = dic_strID[item]
        
        if item in curIvtSet and df_structure[r][i]>df_structure['DPU17_50'][i]: # invest but disrupt, assume after invest, design standard=100 yr
            updateGraph(G, stru_damage, link_damage, item, True)
            computeRepairCost(r, culvertRC, repairC2, 'DPU17_50', item)
            
        elif item not in curIvtSet:
        
            if df_structure['Priority'][i] == 'Low':
                if df_structure[r][i]>df_structure['DPU17_20'][i]:
                    updateGraph(G, stru_damage, link_damage, item, True)
                    computeRepairCost(r, culvertRC, repairC2, 'DPU17_20', item)
            elif df_structure['Priority'][i]=='Medium' or df_structure['Priority'][i]=='High':
                if df_structure[r][i]>df_structure['DPU17_10'][i]:
                    updateGraph(G, stru_damage, link_damage, item, True)
                    computeRepairCost(r, culvertRC, repairC2, 'DPU17_10', item)
            else:
                if df_structure[r][i]>df_structure['DPU17_5'][i]:
                    updateGraph(G, stru_damage, link_damage, item, True)
                    computeRepairCost(r, culvertRC, repairC2, 'DPU17_5', item)                                                                   
                                                          
    
    computeNewGraphCost(dic_graph, 'Culvert', G, baseline, iso, culvertT, isoTrip_sum, num, T, disLoss, demandA_B)
       
    ###################################################################################### repair all        
    for item in cross_set:
        if dic_strNode.get(item,0)==0: continue
        i = dic_strID[item]
        if df_structure[r][i]>df_structure['DPU17_10'][i] or \
            (item not in curIvtSet and df_structure[r][i]>df_structure['DPU17_5'][i]):
                
            updateGraph(G, stru_damage, link_damage, item, True)
            computeRepairCost(r, crossingRC, repairC3, 'DPU17_5', item)
                      
    
    computeNewGraphCost(dic_graph, 'Crossing', G, baseline, iso, crossingT, isoTrip_sum, num, T, disLoss, demandA_B)

    #################################### isolation economic cost

    
    boatCost = baseline*10
    ecoLoss = np.zeros((len(od),len(od)))
    demandAfter = np.zeros((len(od),len(od)))
    demandBefore = np.zeros((len(od),len(od))) 
#    totalC = np.zeros((len(od),len(od))) 
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
            
    # disruption cost, number of disrupted trip, list of damage link, list of damage structure, repair cost of
    # bridge, culvert, crossing, number of isolation trip, change of demand, V: total surplus, isolation cost, isolation demand after alternative mode   
# In[]:     

    # water flood user disruption cost 
    # expected annual user 
    ## fix weird thing in df_structure
for i in range(10):
    df_structure[rperiod[i]] = np.nan_to_num(df_structure[rperiod[i]])
    
wrongInfo=[]    
for i in range(len(df_structure)):
    for j in range(9):
        if df_structure[rperiod[j]][i] > df_structure[rperiod[j+1]][i] :
            wrongInfo.append(i)
for item in wrongInfo:
    for j in range(10):
        df_structure[rperiod[j]][item]=0 
# In[]:            

# calculate the total consumer surplus, V, million$ per year

V=0 # surplus per year (million$)
surplus_d=0 # surplus per day
beta = 2.9
C = lambda x: C_0*np.exp(-1/beta*np.log(x/N_0))    
for j in range(len(od)): # for each OD pair 
    for k in range(j+1,len(od)):
        C_0 = baseCost[j][k]
        N_0 = T[j][k]
        if N_0 !=0:
            surplus = integrate.quad(C, 0, N_0)[0] - C_0 * N_0 # surplus per OD pair
            surplus_d+=surplus
                       
V=surplus_d*365/1e6 

# In[]: Find the maximum repair cost for all infratructure by link when completely destroy

bridgeRC = 13000/1e6 # repair cost $
culvertRC = 2500/1e6
crossingRC = 1250/1e6
maxRC=[] 
dic_linkCost={}   
for i in range(len(df_structure)):
    l = float(df_structure['Length'][i]) 
    if np.isnan(l):
        l=10
    b = float(df_structure['TotalWidth'][i] )
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
# runing sensitivity

from pyDOE import *
from pandas import read_csv,DataFrame,read_stata
import numpy as np


projection1= ['DPU17_5','DPU17_10','DPU17_20','DPU17_50','DPU17_75','DPU17_100','DPU17_200','DPU17_250','DPU17_500','DPU17_1000']   
projection2= ['DPU50_5','DPU50_10','DPU50_20','DPU50_50','DPU50_75','DPU50_100','DPU50_200','DPU50_250','DPU50_500','DPU50_1000']   
projection3= ['DPU00_5','DPU00_10','DPU00_20','DPU00_50','DPU00_75','DPU00_100','DPU00_200','DPU00_250','DPU00_500','DPU00_1000']   
projection = [projection1,projection2,projection3]

baseline = baseCost 

numCases = 50 # number of simulation iteration
stressTest = np.zeros((numCases+1,len(projection)*6)) 
stressPara = pd.DataFrame()
for p in range(3): # three climate projection scenarios
    
    ranges=read_csv("scenarios_ranges.csv") #this contains the min and max value for each uncertain parameter
    numUncertainties=len(ranges)
    lhsample= lhs(numUncertainties,samples=numCases,criterion="corr") # this generates a table with numbers between 0 and 1 (rows are scenarios, columns are the uncertainties)
    lhsample = pd.DataFrame(lhsample)
    scenarios=lhsample.values*np.diff(ranges[['min','max']].values).T + ranges['min'].values # this is to rescale the table to between the min and max of each uncertain parameter
    scenarios_simu=DataFrame(scenarios)
    scenario_base = pd.DataFrame(np.ones((1, numUncertainties)),index=[numCases])
    scenarios = pd.concat([scenarios_simu,scenario_base])
    stressPara=pd.concat([stressPara, scenarios], axis=1)
  
    
    for k in range(numCases+1):
        
        start = time.clock() 
        
        print k
        disUC=[]   
        
        for i in range(10):  
           disUC.append(disrupt(projection[p][i],G2,ivtSet, baseline,scenarios[0][k],scenarios[0][k],scenarios[0][k],scenarios[1][k],scenarios[1][k],scenarios[1][k],T,scenarios[2][k]))
           
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
        avgC =[]        
        numStr = []
        numLink=[]
        
        base = np.sum(np.multiply(baseCost, T))/np.sum(T)
        
        for i in range(10):
            disruptC.append(np.sum(disUC[i][0]))
            isoLoss.append(np.sum(disUC[i][9]))
            disruptT.append(np.sum(disUC[i][1]))
            isolationT.append(np.sum(disUC[i][2]))            
            numLink.append(len(disUC[i][3]))
            numStr.append(len(disUC[i][4]))
            repairC.append((disUC[i][5]+disUC[i][6]+disUC[i][7]))     
            lossTripD.append(np.sum(disUC[i][8]))
        
        
        # find the annual expectation
        EAUC=0 # expected annual disrupt user cost
        isoC=0 # expected annual isolation user cost
        EAR=0  # expected annual repair cost
             
        for i in range(9):                
            EAUC = EAUC + (1.0/rpTime[i]-1.0/rpTime[i+1])*(disruptC[i]+disruptC[i+1])
            isoC = isoC + (1.0/rpTime[i]-1.0/rpTime[i+1])*(isoLoss[i]+isoLoss[i+1])
            EAR = EAR + (1.0/rpTime[i]-1.0/rpTime[i+1])*(repairC[i]+repairC[i+1])
          
        ecoC = EAUC/2+isoC/2
        EARC = EAR/2     
        stressTest[k,6*p] = EARC
        stressTest[k,1+6*p] = ecoC
                  
        print(time.clock() - start, 'seconds')           

    stressTest[:,2+6*p] = stressTest[:,6*p]/sum_RC
    stressTest[:,3+6*p] = stressTest[:,1+6*p]/V       
    stressTest[:,4+6*p] = stressTest[:,6*p]/stressTest[numCases,0]
    stressTest[:,5+6*p] = stressTest[:,1+6*p]/stressTest[numCases,1]

#Output: stressTest: col_1: expected annual repair cost; col_2: expected annual economic loss; 
#col_4: repair cost in Percentage of total maximum asset value, ; col_4: economic loss in percentage of total surplus (dV/V); 
# col_5: ratio to baseline value for repair cost; # ra
st = DataFrame(stressTest)
st.to_csv('Fiji_streeTest_s.csv', sep=',') 
stressPara.to_csv('Fiji_streePara_s.csv', sep=',') 




       