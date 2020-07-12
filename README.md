# enzyme_clustering
The script is designed to perform clustering analysis on molecular dynamics trajectories. 
The descriptors for clustering analysis includes: backbone distances (CA), backbone angles (Phi, Psi, Omega), side chain distances (nearest contacts between heavy atoms), and side chain angles (Chi). 
The first part of the code involves testing on the number of clusters using multiple statistical metrics including inertia, ssh/ssr ratio, etc.
The second part of the code performs clustering on the optimal number of clusters, provides cluster populations, documents the MD snapshot indexes that are closest to the cluster center, plot PCA, and average the structures.

