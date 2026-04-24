#Let's try to teach an elastic network to do non-monotonic curve fitting with its nonlinear allosteric response.
#Start with an elastic network which is a deformed triangular network to remove second-order constraints.
#fix node 0 at the origin and node 2 to lie on the x-axis to remove the trivial translational and rotational degrees of freedom.
#write a relaxation algorithm based on the forces at each node.
#Select a strain uniformly randomly in [0,1].
#strain the nodes out to that strain by stepping slowly in strain, using the relaxation algorithm at each step.
#Use the learning rule to update the spring stiffnesses and/or equilibrium lengths. Then relax again. Repeat this procedure many times.
#At the end, strain back to the starting strain.
#The network may build up pre-stress if the lengths are adjusted, so it may not be fully relaxed at what was originally 0 imposed strain.

import numpy as np
import random
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from scipy.optimize import fsolve

def create_network(L,p,R):
    #Creates a finite, circular network with diameter L, triangular connectivity, and jiggles p in the nodal positions.
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5,np.sqrt(3)/2.0])
    moves = np.array([[(random.random()-0.5)*2*p for i in range(2)] for j in range(2*L**2)])
    nodes = np.array([])
    for xidx in range(L):
        for yidx in range(int((2/np.sqrt(3))*L)):
            node = (xidx-int((L/2)*(1-1/np.sqrt(3)))-np.floor(yidx/2))*a1+(yidx-int((1/np.sqrt(3))*L))*a2 + moves[len(nodes)+1]
            if np.linalg.norm(node)<L/2 and len(nodes)==0:
                nodes = node
            elif np.linalg.norm(node)<L/2:
                nodes = np.vstack((nodes,node)) 
    #construct an incidence matrix based on R. all nodes that are at a distance less than R are connected.
    incidence_matrix = np.array([])
    for i in range(len(nodes)):
        for j in range(i):
            if np.linalg.norm(nodes[i]-nodes[j])<R:
                incidence_matrix_row = np.zeros(len(nodes))
                incidence_matrix_row[j] = 1
                incidence_matrix_row[i] = -1
                if len(incidence_matrix)==0:
                    incidence_matrix = incidence_matrix_row
                else:
                    incidence_matrix = np.vstack((incidence_matrix,incidence_matrix_row))
    #Find good input nodes and rotate the system so that they are horizontally separated
    #The node index where |x|+y is minimal (close to the y axis and towards the bottom).
    in_node_1 = np.where(np.abs(np.transpose(nodes)[0])+np.transpose(nodes)[1]==min(np.abs(np.transpose(nodes)[0])+np.transpose(nodes)[1]))[0][0]
    #Find neighbors of that node, and see which has minimal |x|+y. This pair of nodes will be the input pair.
    bonds = np.where(np.abs(np.transpose(incidence_matrix)[in_node_1])==1)[0]
    nbrs = [np.delete(np.where(np.abs(incidence_matrix[bonds[i]])==1)[0],np.where(np.where(np.abs(incidence_matrix[bonds[i]])==1)[0]==in_node_1)[0][0])[0] for i in range(len(bonds))]
    absxpy = [np.abs(nodes[nbrs[i]][0]) + nodes[nbrs[i]][1] for i in range(len(nbrs))]
    in_node_2 = nbrs[np.where(absxpy==min(absxpy))[0][0]]
    #Make sure that the x coordinate of input node 1 is less than the input coordinate of input node 2.
    if nodes[in_node_1][0]>nodes[in_node_2][0]:
        aux = in_node_1
        in_node_1 = in_node_2
        in_node_2 = aux
    #Pick good output nodes.
    #The node index where |x|-y is minimal (close to the y axis and towards the top).
    out_node_1 = np.where(np.abs(np.transpose(nodes)[0])-np.transpose(nodes)[1]==min(np.abs(np.transpose(nodes)[0])-np.transpose(nodes)[1]))[0][0]
    #Find neighbors of that node, and see which has minimal |x|+y. This pair of nodes will be the input pair.
    bonds = np.where(np.abs(np.transpose(incidence_matrix)[out_node_1])==1)[0]
    nbrs = [np.delete(np.where(np.abs(incidence_matrix[bonds[i]])==1)[0],np.where(np.where(np.abs(incidence_matrix[bonds[i]])==1)[0]==out_node_1)[0][0])[0] for i in range(len(bonds))]
    absxpy = [np.abs(nodes[nbrs[i]][0])-nodes[nbrs[i]][1] for i in range(len(nbrs))]
    out_node_2 = nbrs[np.where(absxpy==min(absxpy))[0][0]]
    #Next we will perform a rotation of the entire system about input node 1 so that input node 2 is purely a horizontal translation away.
    #Every node will be subject to Rotation*(node-node_1)+node_1
    #The rotation is found so that the y coordinate of Rotation*(node_2-node_1)+node_1 is the same as the y coordinate of node_1.
    def Rot(theta):
        return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    def f(theta):
        y = np.sin(theta)*(nodes[in_node_2]-nodes[in_node_1])[0]+np.cos(theta)*(nodes[in_node_2]-nodes[in_node_1])[1]
        return y
    #Find the angle needed to rotate the system so that strains are applied purely horizontally across the input nodes.
    theta = fsolve(f,0)[0]
    for i in range(len(nodes)):
        nodes[i] = Rot(theta)@(nodes[i]-nodes[in_node_1])+nodes[in_node_1]
    #Reconstruct the node list and incidence matrix so that the input nodes are nodes 0 and 1 and the output nodes are 2 and 3.
    nodesnew = nodes[in_node_1]
    nodesnew = np.vstack((nodesnew,nodes[in_node_2]))
    nodesnew = np.vstack((nodesnew,nodes[out_node_1]))
    nodesnew = np.vstack((nodesnew,nodes[out_node_2]))
    rest = [i for i in range(len(nodes)) if i not in [in_node_1,in_node_2,out_node_1,out_node_2]]
    for i in range(len(nodes)-4):
        nodesnew = np.vstack((nodesnew,nodes[rest[i]]))
    nodes = nodesnew
    #reconstruct the incidence matrix
    incidence_matrix = np.array([])
    for i in range(len(nodes)):
        for j in range(i):
            if np.linalg.norm(nodes[i]-nodes[j])<R:
                incidence_matrix_row = np.zeros(len(nodes))
                incidence_matrix_row[j] = 1
                incidence_matrix_row[i] = -1
                if len(incidence_matrix)==0:
                    incidence_matrix = incidence_matrix_row
                else:
                    incidence_matrix = np.vstack((incidence_matrix,incidence_matrix_row))
    #remove the edge between the input nodes.
    incidence_matrix = np.delete(incidence_matrix,(0),axis=0)
    #initialize the stiffnesses to be all ones.
    stiffnesses = np.ones(len(incidence_matrix))
    #determine the equilibrium lengths of all bonds in the network so that it is unstressed at 0 strain.
    eq_lengths = np.linalg.norm(incidence_matrix@nodes,axis=1)
    return nodes, incidence_matrix, eq_lengths, stiffnesses

#Define nodes for task 2, so that we strain them along their initial separation.
def create_network2(nodes):
    def Rot(theta):
        return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    def f2(theta):
        y = np.sin(theta)*(nodes[3]-nodes[2])[0]+np.cos(theta)*(nodes[3]-nodes[2])[1]
        return y
    #Find the angle needed to rotate the system so that strains are applied purely horizontally across the input nodes.
    theta = fsolve(f2,0)[0]
    for i in range(len(nodes)):
        if i==0:
            nodes2 = Rot(theta)@(nodes[i]-nodes[2])+nodes[2]
        else:
            nodes2 = np.vstack((nodes2,Rot(theta)@(nodes[i]-nodes[2])+nodes[2]))
    return nodes2

random.seed(3859823457)
nodes, incidence_matrix, eq_lengths, stiffnesses = create_network(10,0.15,1.6)
#Get stiffnesses from minimization problem.
nodes2=create_network2(nodes)

stiffnesses = np.load('stiffNL4_2_020626.npy')

lines = []
for i in range(len(incidence_matrix)):
    j, k = np.where(np.abs(incidence_matrix[i])>0.5)[0]
    lines.append([(nodes[j,0],nodes[j,1]),(nodes[k,0],nodes[k,1])])

lc = mc.LineCollection(lines, linewidths=5*stiffnesses)
fig, ax = pl.subplots()
ax.add_collection(lc)
plt.scatter(nodes[:,0],nodes[:,1],s=30)
plt.scatter([nodes[0,0],nodes[1,0]],[nodes[0,1],nodes[1,1]],s=30,c='red',zorder=5)
plt.scatter([nodes[2,0],nodes[3,0]],[nodes[2,1],nodes[3,1]],s=30,c='green',zorder=6)
ax.autoscale()
ax.margins(0.1)
ax.set_aspect('equal')

#Imports several auxiliary functions for relaxing the networks.
import functions as f

'''
important functions: 
f.write_lammps_data(filename, positions, incidence, stiffnesses, id_outA=None, id_outB=None, target_output_distance=None, k_output=1e3, mass=1.0)
Writes the lammps data, including an extra extremely stiff spring between the output nodes to enforce the constraint if included.

f.strain_network(datafile, id_fixed, id_pull, clamped = False, dx=0.025, nsteps=200)
Performs the quasistatic strain protocol. The clamped variable determines the name of the bonds file that is referenced.
Outputs relaxed nodal positions after each horizontal strain increment dx.

f.make_video(frames, incidence, stiffnesses, id_fixed, id_pull, id_outA=None, id_outB=None, filename="pulling_network.mp4", interval=50)
Writes videos of strain protocols. Highlights the constrained nodes and bond between them if included.
'''

from matplotlib import colors
import matplotlib as mpl
from matplotlib.colors import ListedColormap
#stiffnesses = np.load('stiffnonrecip2.npy')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'

lines = []
for i in range(len(incidence_matrix)):
    j, k = np.where(np.abs(incidence_matrix[i])>0.5)[0]
    lines.append([(nodes[j,0],nodes[j,1]),(nodes[k,0],nodes[k,1])])

cmapo = mpl.colormaps.get_cmap('Greys')
cmapG = ListedColormap(cmapo(np.linspace(0,1,256)**(1/1.5)))

lc = mc.LineCollection(lines, linewidths=4, array=(stiffnesses), cmap=cmapG)
fig, ax = pl.subplots()
ax.add_collection(lc)
plt.scatter(nodes[:,0],nodes[:,1],s=30,zorder = 5)
plt.scatter([nodes[0,0],nodes[1,0]],[nodes[0,1],nodes[1,1]],s=30,c='red',zorder=5)
plt.scatter([nodes[2,0],nodes[3,0]],[nodes[2,1],nodes[3,1]],s=30,c='green',zorder=6)
plt.xlim((-6,5))
plt.ylim((-5,5))
plt.axis('off')
ax.set_title('Network trained for two tasks',fontsize=18)
cbar = fig.colorbar(lc, ticks=[0.0,0.2,0.4,0.6,0.8,1.0], fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels(['0.0','0.2','0.4','0.6','0.8','1.0'])
cbar.ax.tick_params(labelsize=15)
ax.set_aspect('equal')

strain_input = 0.5
#For LAMMPS protocol (cg)
#final dinputdistance
dinputdistance = (strain_input)*np.linalg.norm(nodes[0]-nodes[1])
#number of steps
nsteps = 1000
#shift per step
dx = dinputdistance/1000
i_din = np.linalg.norm(nodes[0]-nodes[1])
i_dout = np.linalg.norm(nodes[2]-nodes[3])
f.write_lammps_data("data_free.network", nodes, incidence_matrix, stiffnesses)
#Quasistatic shear out to input strain.
nodesf1 = f.strain_network("data_free.network", id_fixed = 0, id_pull = 1, clamped = False, dx=dx, nsteps=nsteps)[nsteps-1]

lines1 = []
for i in range(len(incidence_matrix)):
    j, k = np.where(np.abs(incidence_matrix[i])>0.5)[0]
    lines1.append([(nodesf1[j,0],nodesf1[j,1]),(nodesf1[k,0],nodesf1[k,1])])

cmapo = mpl.colormaps.get_cmap('Greys')
cmapG = ListedColormap(cmapo(np.linspace(0,1,256)**(1/1.5)))

lc1 = mc.LineCollection(lines1, linewidths=4, array=(stiffnesses), cmap=cmapG)
fig, ax = pl.subplots()
ax.add_collection(lc1)
plt.scatter(nodesf1[:,0],nodesf1[:,1],s=30,zorder = 5)
plt.scatter([nodesf1[0,0],nodesf1[1,0]],[nodesf1[0,1],nodesf1[1,1]],s=30,c='red',zorder=5)
plt.scatter([nodesf1[2,0],nodesf1[3,0]],[nodesf1[2,1],nodesf1[3,1]],s=30,c='green',zorder=6)
plt.xlim((-6,5))
plt.ylim((-5,5))
plt.axis('off')
ax.set_title('Network trained for two tasks',fontsize=18)
cbar = fig.colorbar(lc1, ticks=[0.0,0.2,0.4,0.6,0.8,1.0], fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels(['0.0','0.2','0.4','0.6','0.8','1.0'])
cbar.ax.tick_params(labelsize=15)
ax.set_aspect('equal')

strain_input = 1.0
#For LAMMPS protocol (cg)
#final dinputdistance
dinputdistance = (strain_input)*np.linalg.norm(nodes[0]-nodes[1])
#number of steps
nsteps = 1000
#shift per step
dx = dinputdistance/1000
i_din = np.linalg.norm(nodes[0]-nodes[1])
i_dout = np.linalg.norm(nodes[2]-nodes[3])
f.write_lammps_data("data_free.network", nodes, incidence_matrix, stiffnesses)
#Quasistatic shear out to input strain.
nodesf2 = f.strain_network("data_free.network", id_fixed = 0, id_pull = 1, clamped = False, dx=dx, nsteps=nsteps)[nsteps-1]

lines2 = []
for i in range(len(incidence_matrix)):
    j, k = np.where(np.abs(incidence_matrix[i])>0.5)[0]
    lines2.append([(nodesf2[j,0],nodesf2[j,1]),(nodesf2[k,0],nodesf2[k,1])])

cmapo = mpl.colormaps.get_cmap('Greys')
cmapG = ListedColormap(cmapo(np.linspace(0,1,256)**(1/1.5)))

lc2 = mc.LineCollection(lines2, linewidths=4, array=(stiffnesses), cmap=cmapG)
fig, ax = pl.subplots()
ax.add_collection(lc2)
plt.scatter(nodesf2[:,0],nodesf2[:,1],s=30,zorder = 5)
plt.scatter([nodesf2[0,0],nodesf2[1,0]],[nodesf2[0,1],nodesf2[1,1]],s=30,c='red',zorder=5)
plt.scatter([nodesf2[2,0],nodesf2[3,0]],[nodesf2[2,1],nodesf2[3,1]],s=30,c='green',zorder=6)
plt.xlim((-6,5))
plt.ylim((-5,5))
plt.axis('off')
ax.set_title('Network trained for two tasks',fontsize=18)
cbar = fig.colorbar(lc2, ticks=[0.0,0.2,0.4,0.6,0.8,1.0], fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels(['0.0','0.2','0.4','0.6','0.8','1.0'])
cbar.ax.tick_params(labelsize=15)
ax.set_aspect('equal')

import jax.numpy as jnp
#function that constructs the non-extended Hessian given the nodes, stiffnesses, and incidence_matrix.
def buildH(nodes,stiffnesses,incidence_matrix,eq_lengths):
    disps = incidence_matrix@nodes
    #factors, index i
    fs = 1-eq_lengths/np.linalg.norm(disps,axis=1)
    #edge normals, index im
    nhats = np.transpose(np.transpose(disps)/np.linalg.norm(disps,axis=1))
    #off-diagonal Hessian terms, index imn
    odiags1 = jnp.einsum('iimn->imn',jnp.einsum('im,jn',nhats,nhats))
    odiags2 = jnp.einsum('i,imn->imn',fs,jnp.einsum('i,mn->imn',np.ones(len(fs)),np.diag(np.ones(2)))-odiags1)
    odiags = odiags1+odiags2
    H = jnp.einsum('i,ia,ib,imn->ambn',stiffnesses,incidence_matrix,incidence_matrix,odiags)
    return H

#function that constructs the extended Hessian given the nodes, stiffnesses, and incidence_matrix.
#returns only the part that refers to the nodes, not the constraints.
def buildHinvC(nodes,stiffnesses,incidence_matrix,eq_lengths,constrained1,constrained2):
    Pi0 = np.zeros((2,2*incidence_matrix.shape[1]))
    Pi0[0,2*constrained1] = 1
    Pi0[1,2*constrained1+1] = 1
    Pi1 = np.zeros((2,2*incidence_matrix.shape[1]))
    Pi1[0,2*constrained2] = 1
    Pi1[1,2*constrained2+1] = 1
    zeropad = np.array([[0,0],[0,0]])
    H1 = buildH(nodes,stiffnesses,incidence_matrix,eq_lengths)
    Hsupervect = np.reshape(H1,(2*incidence_matrix.shape[1],2*incidence_matrix.shape[1]))
    Hextsupervect = np.block([[Hsupervect,np.transpose(Pi0),np.transpose(Pi1)],
                              [Pi0,      zeropad,           zeropad],
                              [Pi1,      zeropad,           zeropad]])
    Hextinvsupervect = np.linalg.inv(Hextsupervect)[:-4,:-4]
    Hextinvclipped = np.reshape(Hextinvsupervect,(incidence_matrix.shape[1],2,incidence_matrix.shape[1],2))
    return Hextinvclipped

#this will build - si ML si.
def constructLong(nodes,stiffnesses,incidence_matrix,eq_lengths,Hinv):
    nhats = jnp.einsum('ia,am',incidence_matrix,nodes)
    nhats = np.transpose(np.transpose(nhats)/np.linalg.norm(nhats,axis=1))
    dHinvpardki = -1.0*jnp.einsum('amcr,ic,ir,ig,id,dgbn->iambn',Hinv,incidence_matrix,nhats,nhats,incidence_matrix,Hinv)
    return dHinvpardki
    
#this will build -si MT si
def constructTrans(nodes,stiffnesses,incidence_matrix,eq_lengths,Hinv):
    nhats = jnp.einsum('ia,am',incidence_matrix,nodes)
    nhats = np.transpose(np.transpose(nhats)/np.linalg.norm(nhats,axis=1))
    f = stiffnesses*(1-eq_lengths/np.linalg.norm(incidence_matrix@nodes,axis=1))
    kd = np.array([[1,0],[0,1]])
    dHinvperpdki1 = -1.0*jnp.einsum('i,amcr,ic,rg,id,dgbn->iambn',f,Hinv,incidence_matrix,kd,incidence_matrix,Hinv)
    dHinvperpdki2 = jnp.einsum('i,amcr,ic,ir,ig,id,dgbn->iambn',f,Hinv,incidence_matrix,nhats,nhats,incidence_matrix,Hinv)
    return dHinvperpdki1+dHinvperpdki2

#this will build fi sum_j(sj Gij sj)
def constructEq(nodes,stiffnesses,incidence_matrix,eq_lengths,Hinv):
    nhats = jnp.einsum('ia,am',incidence_matrix,nodes)
    nhats = np.transpose(np.transpose(nhats)/np.linalg.norm(nhats,axis=1))
    f = stiffnesses*(1-eq_lengths/np.linalg.norm(incidence_matrix@nodes,axis=1))
    kd = np.array([[1,0],[0,1]])
    ells = np.linalg.norm(incidence_matrix@nodes,axis=1)
    sijmn = jnp.einsum('ia,ambn,jb->ijmn',incidence_matrix,Hinv,incidence_matrix)
    ones = np.ones(len(f))
    Gt1 = jnp.einsum('j,i,j,jn,jimg,ig->ijmn',ones-f,ells,1/ells,nhats,sijmn,nhats)
    Gt2 = jnp.einsum('j,i,j,jm,jing,ig->ijmn',ones-f,ells,1/ells,nhats,sijmn,nhats)
    Gt3 = jnp.einsum('j,i,j,jb,jibg,ig,mn->ijmn',ones-f,ells,1/ells,nhats,sijmn,nhats,kd)
    Gt4 = -3.0*jnp.einsum('j,i,j,jb,jibg,ig,jm,jn->ijmn',ones-f,ells,1/ells,nhats,sijmn,nhats,nhats,nhats)
    G = Gt1 + Gt2 + Gt3 + Gt4
    #dHdk
    internal = -1.0*jnp.einsum('i,j,ja,ijmn,jb->iambn',f,stiffnesses,incidence_matrix,G,incidence_matrix)
    dHinveqdki = -1.0*jnp.einsum('amcr,icrdg,dgbn->iambn',Hinv,internal,Hinv)
    return dHinveqdki


Hinv0 = buildHinvC(nodes,stiffnesses,incidence_matrix,eq_lengths,0,1)
Hinv1 = buildHinvC(nodesf1,stiffnesses,incidence_matrix,eq_lengths,0,1)
Hinv2 = buildHinvC(nodesf2,stiffnesses,incidence_matrix,eq_lengths,0,1)

dHinvpardkint = constructLong(nodes,stiffnesses,incidence_matrix,eq_lengths,Hinv0)
spar0 = -jnp.einsum('iamam->i',dHinvpardkint)
dHinvpardkint = constructLong(nodesf1,stiffnesses,incidence_matrix,eq_lengths,Hinv1)
spar1 = -jnp.einsum('iamam->i',dHinvpardkint)
dHinvpardkint = constructLong(nodesf2,stiffnesses,incidence_matrix,eq_lengths,Hinv2)
spar2 = -jnp.einsum('iamam->i',dHinvpardkint)

dHinvpardkint = constructTrans(nodes,stiffnesses,incidence_matrix,eq_lengths,Hinv0)
sperp0 = -jnp.einsum('iamam->i',dHinvpardkint)
dHinvpardkint = constructTrans(nodesf1,stiffnesses,incidence_matrix,eq_lengths,Hinv1)
sperp1 = -jnp.einsum('iamam->i',dHinvpardkint)
dHinvpardkint = constructTrans(nodesf2,stiffnesses,incidence_matrix,eq_lengths,Hinv2)
sperp2 = -jnp.einsum('iamam->i',dHinvpardkint)

dHinvpardkint = constructEq(nodes,stiffnesses,incidence_matrix,eq_lengths,Hinv0)
seq0 = -jnp.einsum('iamam->i',dHinvpardkint)
dHinvpardkint = constructEq(nodesf1,stiffnesses,incidence_matrix,eq_lengths,Hinv1)
seq1 = -jnp.einsum('iamam->i',dHinvpardkint)
dHinvpardkint = constructEq(nodesf2,stiffnesses,incidence_matrix,eq_lengths,Hinv2)
seq2 = -jnp.einsum('iamam->i',dHinvpardkint)