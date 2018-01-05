# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:58:41 2018

@author: Samqua
@version: 1.0

A simulated annealing Monte Carlo optimization algorithm for cost functions defined on simple graphs of order N.
Specify the initial coordinates of the nodes (or randomize them) inside a square of arbitrary side length.
The number of simple graphs of order N>20 is an incomprehensibly large number.
The problem of finding the unique graph that optimizes a cost function is therefore computationally intractable if N is large.
Instead, we aspire merely to a near-optimal graph using a probabilistic search algorithm.
The algorithm, denoted perturb() in the code, tweaks a graph until it finds one of higher fitness, and then caches it.
It jumps randomly through the network state space in an attempt to avoid getting stuck in a local fitness maximum.

It does, however, have a nonzero chance (denoted K in the code) to return to the current optimum.
The average radius of the algorithm's orbit in state space is controlled by this parameter K:
- if K is too high, the algorithm will be constrained to a local fitness maximum.
- if K is too low, we could potentially be spending too much time searching an extraneous region of state space.
So there is a goldilocks range of K values for which the algorithm finds a good solution quickly.

The fitness function is defined node-wise: every node has a fitness, and we judge graphs by the sum of all node fitnesses.
Contributions to fitness include:

(1) Degree (total number of connections)
(2) Domination of neighbors (having more connections than the average of those connected to it)
(3) Minimizing the sum of edge costs, where cost is proportional to edge length

Parameters p1, p2, and p3 respectively control the relative weights of these considerations in the fitness function.

"""
#####################################################

import random
#import numpy as np
#import matplotlib.pyplot as plt
import plotly as py
from plotly.graph_objs import *
import networkx as nx
import copy
import datetime

#####################################################

start=datetime.datetime.now()
print("START: "+str(start))

side_length=1000
total_population=50000 # currently unused
N=100 # try not to exceed N=300
steps=500 # temperature resolution
tries=1 # number of times to repeat entire algorithm
max_degree=5 # possibly unused
K=0.0001 # small chance to return us to the current fitness maximum, recommended 0.0001
p1=250 # 250 recommended
p2=300 # 300 recommended
p3=1 # 1 recommended

init_coord=[]
for i in range(0,N):
        init_coord.append([random.uniform(1,side_length+1),random.uniform(1,side_length+1)]) # coordinates allowed to be floats; ints can be obtained using ceiling and floor functions

"""
Example data:
init_coord=[[581.791426227322, 980.2823816645615], [204.75003029267714, 399.47965394745523], [734.2910136543942, 959.272398908303], [482.25541188396915, 407.04404493361267], [454.07934038650365, 993.1681823913875], [494.08133543063724, 201.10639908417593], [190.2904832122998, 378.86241146040334], [443.2660649398715, 627.9892016440075], [254.31362954039884, 919.7602962897885], [669.8892302824736, 746.7185902817512], [100.98426350267509, 916.9395915199017], [596.03385724617, 520.4959985521325], [792.5633043803628, 987.4758613729537], [583.6409197972246, 984.9372327761208], [403.9033730058758, 94.64920761501705], [430.24387295293144, 570.8566632868902], [956.7953867526576, 169.7157189341898], [154.4125875017839, 835.0229998213775], [983.422336564179, 335.0586078446084], [754.7434474348257, 526.6270406940486], [893.41566910758, 124.76590866603077], [445.87255944181027, 279.5133187847811], [232.69663703783883, 97.89256958226011], [830.9529511235411, 442.9650809587594], [613.1487038507444, 133.26492951823298], [700.7472233184681, 613.1136190857171], [556.9879495256503, 78.0549831438918], [628.7951843969191, 59.10891920617423], [421.78943283603667, 440.84608112801817], [365.39017591005506, 989.1498689065212], [566.9042846762642, 984.5661169804623], [555.5818251672656, 529.9485169836548], [755.170945655323, 261.3577815224904], [949.8184671843018, 330.52748482601936], [289.03985698616594, 973.597150640252], [581.0810685071121, 312.524755268291], [585.1067198725104, 953.4133094801513], [57.244769642270434, 42.847632693124524], [747.1796601174569, 536.4722876639764], [685.5834866783753, 922.7983334511849], [186.40836703322162, 80.86431504489732], [230.30271698340775, 391.7168356248111], [655.6620130216081, 768.1430685605403], [664.7939120640195, 273.20772687995475], [659.034314341772, 623.6189550417979], [1.0033010907031228, 448.2725476551921], [647.6898038430311, 970.8092041500124], [587.3820733771083, 456.0967742069174], [371.41851049805643, 303.043045852188], [197.0039553870556, 814.8833762160416], [723.1015299593533, 660.2079594435418], [665.3758928359113, 497.523982993188], [747.7998236852278, 924.0968281624811], [753.7781569808496, 497.5651428463399], [664.6573828656484, 46.34453816119044], [23.583270750772154, 834.85660139277], [155.43964588414028, 280.5883076865755], [661.622830848578, 342.6593234321673], [911.6139284586251, 845.5677141288219], [120.31872140107203, 846.3238412423677], [475.19280864187596, 854.1927651876649], [443.38868763856675, 776.0780419205204], [702.5350896648093, 340.81196131246816], [336.9552484975108, 792.8784873223713], [413.92723367490515, 448.77734581211234], [16.569443089199385, 219.90714059648252], [757.2288768732669, 632.6557600111852], [388.1248177977862, 767.0279968711139], [223.28552453219984, 7.118031533171808], [106.12824430883228, 758.357145063579], [949.7020774666419, 604.51546054644], [83.99080877820653, 454.94283901402264], [962.6367699059444, 189.90207785008494], [994.1166296636725, 155.2380336136797], [336.75548394517665, 44.231549405532974], [965.5680936064116, 879.9039833418105], [377.62059342523366, 454.17996138577263], [645.6531403029445, 669.3978978122287], [282.0454448173506, 875.2076352672058], [313.3097394120724, 144.0201668677108], [39.67497949715537, 770.7472296030677], [979.0039479519219, 948.7349513148849], [123.65091366977477, 73.92129131690439], [677.6359560335701, 432.0458842546909], [756.936804683188, 706.1153347902326], [159.75668571349178, 425.3196805549564], [763.5779338253888, 265.5928375486345], [376.98362924549235, 923.2191299939943], [760.5327405830503, 508.92141824153424], [749.77810497677, 733.1829905321282], [506.50534325594873, 51.03647622380336], [694.2104207443042, 494.77777942327606], [525.7589148350738, 531.4187645769891], [844.5045546720521, 453.55945398837747], [398.4682025771131, 787.5361602580505], [962.3085160368572, 58.22275150103273], [260.0385049166625, 492.06537569749366], [161.32611346891446, 7.71372363368572], [184.6148488527195, 227.53501715450187], [173.10596221479037, 465.85734912991825]]
"""

#print(init_coord)
#print("\n")

#####################################################

def distance(ax,ay,bx,by):
        return ((abs(bx-ax))**2+(abs(by-ay))**2)**0.5

distance_matrix = [[0 for col in range(N)] for row in range(N)] # initialize distance matrix
from_bottom_left = [0] * N

for i in range(0,N):
        from_bottom_left[i] = init_coord[i][0]+init_coord[i][1]
    
nodes = {} #intialize labels

for i in range(0, N):
        nodes[i]=copy.deepcopy(init_coord[from_bottom_left.index(sorted(from_bottom_left)[i])]) # n.b. these are static labels and will not change from here on out

pos=copy.deepcopy(nodes)

for i in range(0,N):
        for j in range(0,N):
                distance_matrix[i][j]=distance(nodes[i][0],nodes[i][1],nodes[j][0],nodes[j][1]) #create symmetric distance matrix where (i,j) is the distance from j to i. note zeros along the main diagonal

distance_dict={}
for i in nodes:
        for j in nodes:
                distance_dict[(i,j)]=distance_matrix[i][j]

#####################################################

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

#####################################################

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

#####################################################

def doesEdgeExist(i,j): #does an edge exist from node i to node j
        if j in graph[i]:
                return True
        else:
                return False

def step(x):
        return 1*(x>0)

def deg(i): # returns degree of node i
        return len(graph[i])

def total_degree():
        z=0
        for i in range(0,N):
                z+=deg(i)
        return z

def dom_thresh(i): # returns the degree threshold node i must exceed to dominate its neighbors
        if deg(i)==0:
                return 1
        else:
                z=0
                for j in graph[i]:
                        z+=deg(j)
                z=z/deg(i)
                return z

def edge_cost(i,j): # returns the cost of the edge between i and j
        if doesEdgeExist(i,j)==False:
                return 0 # if the edge doesn't exist, the cost is 0
        else:
                return distance_matrix[i][j]

def total_edge_cost(i):
        z=0
        for j in graph[i]:
                z+=edge_cost(i,j)
        return z
    
def fitness(i):
        return p1*deg(i)+p2*(deg(i)-dom_thresh(i))-p3*total_edge_cost(i) # currently fitness does not depend on population

def total_fitness():
        z=0
        for j in range(0,N):
                z+=fitness(j)
        return z

#####################################################

for i in nodes:
        nodes[i].append(total_population/N)
        nodes[i].append(fitness(i)) # nodes[i][0] is x-coord, nodes[i][1] is y-coord, nodes[i][2] is population, nodes[i][3] is fitness
        
def population(i): # returns current population of node i
        return nodes[i][2]

best_fitness=total_fitness()
best_graph=copy.deepcopy(graph)

#####################################################

graph_count=1

def perturb(desiredsteps=100):
        for i in range(0,N):
                for j in range(0,N): #findNearest(i,max_degree-1) #range(0,N)
                        PC = random.random()
                        PD = random.random()
                        PK = random.random()
                        global temp
                        global graph_count
                        global graph
                        d=distance_matrix[i][j]
                        C=temp*step((300-0.73*N)-d)*(1-(d/1415))**3
                        D=temp*step(300-0.73*N)*(d/1415)**0.2
                        if PK <= K:
                                graph=copy.deepcopy(best_graph)
                        if PC <= C and i != j and doesEdgeExist(i,j) == False:
                                graph[i]=graph[i]+[j]
                                graph[j]=graph[j]+[i]
                                graph[i].sort()
                                graph[j].sort()
                        if PD <= D and i != j and doesEdgeExist(i,j) == True:
                                graph[i]=[x for x in graph[i] if x != j]
                                graph[j]=[x for x in graph[j] if x != i]
                        temp=temp-(1/(N*N*desiredsteps))
                        graph_count+=1

#####################################################

for i in range(tries):
        temp=1
        while temp > 0:
                perturb(steps)
                if total_fitness()>best_fitness:
                        best_fitness=total_fitness()
                        best_graph=copy.deepcopy(graph)

if best_fitness>100000:
        print("\n")
        print(best_graph)
        print("\n")

#####################################################

Gf=nx.Graph(graph)
nx.set_node_attributes(Gf,pos,'pos')

G=nx.Graph(best_graph)
nx.set_node_attributes(G,pos,'pos')

#####################################################

def best_total_degree():
    z=0
    for i in best_graph:
        z+=len(best_graph[i])
    return z

#####################################################

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
                title='<br>Network, N='+str(N)+", P1="+str(p1)+" , P2="+str(p2)+", P3="+str(p3)+", K="+str(K)+", Tries: "+str(tries)+", Temperature resolution: "+str(steps)+", Fitness: "+str(best_fitness),
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

py.offline.plot(fig, filename='net_sim '+datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")+'.html')

print("END: "+str(datetime.datetime.now()))
print("RUN TIME: "+str(datetime.datetime.now()-start))
print("\n")
print("NODES: "+str(N)+" | TRIES: "+str(tries)+" | TEMPERATURE RESOLUTION: "+str(steps))
print("TOTAL GRAPHS PROCESSED: "+str(graph_count))
print("P1: "+str(p1)+" | P2: "+str(p2)+" | P3: "+str(p3)+" | K: "+str(K))
print("FINAL | "+"AVG DEGREE: "+ str(total_degree()/N)+" | TOTAL CYCLES: "+str(len(nx.cycle_basis(Gf))))
print("BEST | "+"AVG DEGREE: "+ str(best_total_degree()/N)+" | TOTAL CYCLES: "+str(len(nx.cycle_basis(G))))
print("     | CYCLES PER NODE: "+str(len(nx.cycle_basis(G))/N))