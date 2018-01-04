# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 23:56:59 2018

A simulated annealing Monte Carlo method to optimize a cost function (denoted fitness in the code) defined on a network of N nodes.

@author: Samqua
github.com/Samqua

"""

import random
#import numpy as np
#import matplotlib.pyplot as plt
import plotly as py
from plotly.graph_objs import *
import networkx as nx
import copy
import datetime

N=100 # try not to exceed N=300
steps=500 # temperature resolution
tries=1 # number of times to repeat entire algorithm
max_degree=5 #currently unused

init_coord=[]
for i in range(0,N):
    init_coord.append([random.randrange(1,1001),random.randrange(1,1001)])

temp=1.0 #initialize temperature
init_fitness=-99999999 #initial fitness is assumed to be a large negative value
best_fitness=init_fitness

#data = np.array(init_coord)
#x, y = data.T
#plt.scatter(x,y)

def distance(ax,ay,bx,by):
    return ((abs(bx-ax))**2+(abs(by-ay))**2)**0.5

distance_matrix = [[0 for col in range(N)] for row in range(N)] #initialize distance matrix

from_bottom_left = [0] * N #initialize from_bottom_left

for i in range(0,N):
    from_bottom_left[i] = init_coord[i][0]+init_coord[i][1]
    
towns = {} #intialize labels

for i in range(0, N):
    towns[i]=init_coord[from_bottom_left.index(sorted(from_bottom_left)[i])] #n.b. these are static labels and will not change from here on out

for i in range(0,N):
    for j in range(0,N):
        distance_matrix[i][j]=distance(towns[i][0],towns[i][1],towns[j][0],towns[j][1]) #create symmetric distance matrix where (i,j) is the distance from j to i. note zeros along the main diagonal

s=[]
for i in range(0,N):
    for j in range(0,N):
        if i!=j:
            s.append(distance_matrix[i][j])
print("AVERAGE DISTANCE BETWEEN POINTS: "+str(sum(s)/len(s))) # should be around 500 by default, or whatever the side length is divided by 2

normalized_towns = copy.deepcopy(towns)

for i in towns:
    normalized_towns[i][0]=towns[i][0]/1000
    normalized_towns[i][1]=towns[i][1]/1000

C=0.5 #initialize C
D=0.5 #initialize D

distance_dict={}
for i in towns:
    for j in towns:
        distance_dict[(i,j)]=distance_matrix[i][j]

def findNearest(i,n): # returns a list of labels of the n nodes nearest node i
    d=[]
    for j in range(0,N):
        d.append([(i,j),distance_dict[(i,j)]])
    d=sorted(d, key=lambda a: a[1])
    d=d[1:n+1]
    q=[]
    for i in range(0,len(d)):
        q.append(d[i][0][1])
    return q

"""
graph = {0:[1],(N-1):[(N-2)]} #initialize graph

for i in range(1,N-1):
    graph[i]=[i-1,i+1] #initial guess for graph
"""

graph={0:[1]}
for i in range(1,N):
    graph[i]=findNearest(i,2)

for i in range(0,N):
    for j in range(0,N):
        if i in graph[j] and j not in graph[i]:
            graph[i].append(j)
    if i in graph[i]:
        graph[i].remove(i)
    graph[i].sort()

best_graph=copy.deepcopy(graph)

def doesRoadExist(i,j): #does a road exist from i to j
    if j in graph[i]:
        return True
    else:
        return False

def step(x):
    return 1*(x>0)

def perturb(desiredsteps=100):
    for i in range(0,N):
        for j in range(0,N): #findNearest(i,max_degree-1)
            PC = random.random()
            PD = random.random()
            global temp
            d=distance(towns[i][0],towns[i][1],towns[j][0],towns[j][1])
            C=temp*step((300-0.73*N)-d)*(1-(d/1415))**3
            D=temp*step(300-0.73*N)*(d/1415)**0.2
            if PC <= C and i != j and doesRoadExist(i,j) == False:
                graph[i]=graph[i]+[j]
                graph[j]=graph[j]+[i]
                graph[i].sort()
                graph[j].sort()
            if PD <= D and i != j and doesRoadExist(i,j) == True:
                graph[i]=[x for x in graph[i] if x != j]
                graph[j]=[x for x in graph[j] if x != i]            
            temp=temp-(1/(N*N*desiredsteps))

def total_distance():
    total_sum_of_distances=0
    for i in graph:
        sum_of_distances=0
        for j in graph[i]:
            sum_of_distances+=distance(towns[i][0],towns[i][1],towns[j][0],towns[j][1])
        total_sum_of_distances+=sum_of_distances
    return total_sum_of_distances

def total_degree():
    td=0
    for i in graph:
        td+=len(graph[i])
    return td

def fitness():
    for i in graph:
        if graph[i]==[]:
            return init_fitness
    return -total_distance()*total_degree()

print("START: "+str(datetime.datetime.now()))

for i in range(tries):
    temp=1
    while temp > 0:
        perturb(steps)
        if fitness()>best_fitness:
            best_fitness=fitness()
            best_graph=copy.deepcopy(graph)

print("END: "+str(datetime.datetime.now()))

def best_total_degree():
    td=0
    for i in best_graph:
        td+=len(best_graph[i])
    return td

Gf=nx.Graph(graph)
pos=normalized_towns
nx.set_node_attributes(Gf,pos,'pos')

G=nx.Graph(best_graph)
nx.set_node_attributes(G,pos,'pos')

print("NODES: "+str(N)+" | TRIES: "+str(tries)+" | TEMPERATURE RESOLUTION: "+str(steps))
print("FINAL | "+"AVG DEGREE: "+ str(total_degree()/N)+" | TOTAL CYCLES: "+str(len(nx.cycle_basis(Gf))))
print("BEST | "+"AVG DEGREE: "+ str(best_total_degree()/N)+" | TOTAL CYCLES: "+str(len(nx.cycle_basis(G))))

dmin=1
ncenter=0
for n in pos:
    x,y=pos[n]
    d=(x-0.5)**2+(y-0.5)**2
    if d<dmin:
        ncenter=n
        dmin=d

p=nx.single_source_shortest_path_length(G,ncenter)

edge_trace = Scatter(
    x=[],
    y=[],
    line=Line(width=0.5,color='#888'),
    hoverinfo='none',
    mode='lines')

for edge in G.edges():
    x0, y0 = G.node[edge[0]]['pos']
    x1, y1 = G.node[edge[1]]['pos']
    edge_trace['x'] += [x0, x1, None]
    edge_trace['y'] += [y0, y1, None]

node_trace = Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=Marker(
        showscale=True,
        # colorscale options
        # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
        # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
        colorscale='Hot',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line=dict(width=2)))

for node in G.nodes():
    x, y = G.node[node]['pos']
    node_trace['x'].append(x)
    node_trace['y'].append(y)

for i,j in list(nx.degree(G)):
    node_trace['marker']['color'].append(j)
    node_info = 'Node: '+str(i)+', Degree: '+str(j)
    node_trace['text'].append(node_info)


fig = Figure(data=Data([edge_trace, node_trace]),
             layout=Layout(
                title='<br>Road Network, N='+str(N),
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code modified from Plotly example",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

py.offline.plot(fig, filename='sim-anneal.html')
