# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 23:56:59 2018

@author: Samqua
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import plotly as py
from plotly.graph_objs import *
import networkx as nx
import copy

N=120 # no edges will form if N exceeds 300

init_coord=[]
for i in range(0,N):
    init_coord.append([random.randrange(1,1001),random.randrange(1,1001)])

temp=1.0 #initialize temperature

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
print("Average distance:")
print(sum(s)/len(s))
        
        
graph = {0:[1],(N-1):[(N-2)]} #initialize graph

for i in range(1,N-1):
    graph[i]=[i-1,i+1] #initial guess for graph

def doesRoadExist(i,j): #does a road exist from i to j
    if j in graph[i]:
        return True
    else:
        return False

C=0.5 #initialize C
D=0.5 #initialize D


def step(x):
    return 1*(x>0)

def perturb(desiredsteps=10000.0):
    for i in range(0,N):
        for j in range(0,N):
            PC = random.random()
            PD = random.random()
            global temp
            d=distance(towns[i][0],towns[i][1],towns[j][0],towns[j][1])
            C=temp*step((300-0.73*N)-d)*(1-(d/1415))**3
            D=temp*step((300-0.73*N))*(d/1415)**0.2
            if PC <= C and i != j and doesRoadExist(i,j) == False:
                graph[i]=graph[i]+[j]
                graph[j]=graph[j]+[i]
                graph[i].sort()
                graph[j].sort()
            if PD <= D and i != j and doesRoadExist(i,j) == True:
                graph[i]=[x for x in graph[i] if x != j]
                graph[j]=[x for x in graph[j] if x != i]
            temp=temp-(1/(N*N*desiredsteps))

while temp > 0.01:
    perturb(100)

normalized_towns = copy.deepcopy(towns)

for i in towns:
    normalized_towns[i][0]=towns[i][0]/1000
    normalized_towns[i][1]=towns[i][1]/1000

G=nx.Graph(graph)
pos=normalized_towns
nx.set_node_attributes(G,pos,'pos')

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
    node_info = 'Node: '+str(i)+', Degree: '+str(j) # ayy
    node_trace['text'].append(node_info)


fig = Figure(data=Data([edge_trace, node_trace]),
             layout=Layout(
                title='<br>Road Network',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

py.offline.plot(fig, filename='sim-anneal.html')
