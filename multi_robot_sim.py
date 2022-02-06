#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 19:33:15 2022

@author: bdobkowski
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import matplotlib.animation as animation
from tqdm import tqdm

# Dim of state space for each robot
dim = 2
deterministic_graph = False

# Proximity for graph edge
R   = 5

y    = np.loadtxt('./x2.txt')
temp = 10 - y
y    = y + 2*temp
x    = np.loadtxt('./y2.txt')

n         = len(x)
neighbors = {}
x_star    = np.zeros((n,n,2))
d_matrix  = np.zeros((n,n))

states = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)

def lcp(x, t):
    '''
    LCP ODE for the odeint solver. This function loops through the neighbors
    for each node, and computes the linear consensus protocol control law.
    '''
    # Calling graph maker based on proximity between nodes
    
    x = x.reshape(n, dim)
    
    ans = np.zeros((n, dim))
            
    # Control Law
    for i in range(n):
        if i in list(neighbors.keys()):
            for j in neighbors[i]:
                
                # Nonlinear formation controller
                ans[i] += ((x[j] - x[i]).T @ (x[j] - x[i]) - d_matrix[i,j])*(x[j] - x[i])
                
                # Affine formation controller
                # ans[i] += x[j] - x[i] - x_star[i,j]               
                
    return ans.flatten()

def create_proximity_graph(x):
    '''
    Creates a topological graph for the given number of robots, n based on the
    proximity of robot i to robot j. An edge is only added between nodes if
    the norm between states i and j is <= R.
    '''
    L = np.zeros((n, n))
    neighbors = {}
    D = np.zeros((n, n))
    A = np.zeros((n, n))    
    edges = []
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            if np.linalg.norm(x[i] - x[j]) <= R:
                x_star[i,j,:] = x[j] - x[i]
                d_matrix[i, j] = np.linalg.norm(x[j] - x[i])**2
                if not [i, j] in edges and not [j, i] in edges:
                    edges.append([i, j])
                    
    for i in range(n):
        neighborlist = []
        for j in edges:
            if i in j:
                if j[0] == i:
                    neighborlist.append(j[1])
                    A[i, j[1]] = 1
                else:
                    neighborlist.append(j[0])
                    A[i, j[0]] = 1
        neighbors[i] = neighborlist
        D[i, i] = len(neighborlist)
    L = D - A
    assert np.linalg.matrix_rank(L) >= n-1
    return neighbors, L

neighbors, L = create_proximity_graph(states)

# t   = np.linspace(0,1,801)      
t   = np.linspace(0,2,51)                
x0  = np.max([np.max(x),np.max(y)])*np.random.rand(n, dim)
print('solving ODE')
sol = odeint(lcp, x0.flatten(), t)   
print('done solving ODE') 
sol = sol.reshape(-1, n, dim)

fig, ax = plt.subplots(dim)
fig.suptitle('Affine Formation Cont for {} Agents'.format(n))

for d in range(dim):
    ax[d].plot(np.ones(len(t))*np.mean(x0[:,d]),'k:')
    ax[d].plot(sol[:, :, d])
    ax[d].set_xlabel('Time Steps')
    ax[d].set_ylabel('Scalar Pos - Dim {}'.format(d))
    ax[d].legend(['Avg of Initial Conditions'])
plt.savefig('plot1csmallEdge.png', bbox_inches='tight')
    
fig1, ax1 = plt.subplots()
ax1.scatter(sol[0, :, 0], sol[0, :, 1], c='k', label='Initial Pos')
ax1.scatter(sol[-1, :, 0], sol[-1, :, 1], c='r', label='Final Pos')
ax1.plot(sol[:, :, 0], sol[:, :, 1])
ax1.set_title('2-Dim Plot of Robot Trajectories')
ax1.set_xlabel('X Position (Dim 0)')
ax1.set_ylabel('Y Position (Dim 1)')
ax1.legend()
plt.savefig('plot2csmallEdge.png', bbox_inches='tight')

fig2, ax2 = plt.subplots()
ax2.scatter(sol[-1, :, 0], sol[-1, :, 1], c='r', label='Final Pos')
ax2.set_title('2-Dim Plot of Robot Trajectories')
ax2.set_xlabel('X Position (Dim 0)')
ax2.set_ylabel('Y Position (Dim 1)')
ax2.legend()
plt.savefig('plot2ccsmallEdge.png', bbox_inches='tight')
plt.show()

# fig3, ax3 = plt.subplots()
# lns = []
# for i in tqdm(range(len(sol))):
#     ln  = ax3.scatter(sol[i,:,0], sol[i,:,1], c='r')
#     tm  = ax3.text(-1, 0.9, 'time = %.2fs' % t[i])
#     lns.append([ln, tm])
# ax3.set_aspect('equal', 'datalim')
# ax3.grid()
# ani = animation.ArtistAnimation(fig3, lns, repeat=True, interval=1500)
# ani.save('./animation.mp4',writer='ffmpeg',fps=1000/50)
# # ani.save('./animation.gif',writer='imagemagick',fps=1000/50)


            
