import mdtraj as md
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import preprocessing
from msmbuilder.cluster import KMeans
from itertools import combinations
from sklearn.model_selection import learning_curve, GridSearchCV
from scipy.spatial import distance
import seaborn as sns
import numpy as np

def ssr_sst_ratio(X, labels):
    """adapted from the Calinski and Harabasz score.
    """
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    extra_disp, intra_disp = 0., 0.
    mean = np.mean(X, axis=0)
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)

    return (1. if intra_disp == 0. else
            extra_disp / (intra_disp+extra_disp))

def Kmeans_score(dataset, Max_clusters):
    print("Start to analyze the dependence of inertia on the number of clusters\n")
    scores_in = [] #the elbow indicates the good cluster number
    scores_sc = [] #s=b-a/max(a,b) a: The mean distance between a sample and all other points in the same class,
    # b: The mean distance between a sample and all other points in the next nearest cluster.
    scores_ch = [] # Variance Ratio Criterion, tightness of the cluster
    scores_rt = [] # Variance ratio, As the ratio inherently rises with cluster count,
    # one looks for an “elbow” in the curve where adding another cluster does not add much new information, as done in a scree test
    scores_db = [] # Values closer to zero indicate a better partition. sum of cluster i and j diameter over the distance between cluster centroids i and j. smaller the better.
    for i in range(Max_clusters - 2):
        kmeans_model = KMeans(n_clusters=i+2, init='k-means++', n_init=10, max_iter=300, tol=0.001,
                     precompute_distances='auto', verbose=0, random_state=None,
                     copy_x=True,n_jobs=1).fit(dataset)
        labels = kmeans_model.labels_
        scores_in.append(kmeans_model.inertia_)
        scores_sc.append(metrics.silhouette_score(dataset[0], labels[0], metric='euclidean'))
        scores_ch.append(metrics.calinski_harabaz_score(dataset[0], labels[0]))
        scores_rt.append(ssr_sst_ratio(dataset[0], labels[0]))
        scores_db.append(metrics.davies_bouldin_score(dataset[0], labels[0]))
    print("Done generating scores for "+str(Max_clusters)+" clusters\n")
    return scores_in, scores_sc, scores_ch, scores_rt, scores_db

def Plot_scores(Max_clusters,scores_in,name):
    plt.subplots()
    sns1=sns.lineplot(np.arange(2, Max_clusters), scores_in)
    plt.xlabel('Number of clusters')
    plt.ylabel(name)
    plt.title("Inertia of k-Means versus number of clusters")
    plt.savefig('./AlleyCat-Ca-unconstrained/'+name+'.pdf', dpi=300)
    print("Done plotting for "+name)
    return 1

def dataset_CA_distances(traj):
    dataset = []
    topo = traj.topology
    A = [atom.index for atom in topo.atoms if (atom.name == 'CA')]
    comb = [subset for subset in combinations(np.arange(len(A)), 2)]
    traj_subset = traj.atom_slice(A)
    print("Done constructing dataset using CA distances")
    dataset.append(md.compute_distances(traj_subset, atom_pairs=comb))
    return dataset

def dataset_phi_psi_omega(traj):
    dataset = []
    indices,angles1 = md.compute_phi(traj)
    indices,angles2 = md.compute_psi(traj)
    indices,angles3 = md.compute_omega(traj)
    angles = np.concatenate((angles1,angles2,angles3), axis=1)
    dataset.append(angles)
    print("Done constructing dataset using Phi angles")
    return dataset

def dataset_chi(traj):
    dataset = []
    indices,angles1 = md.compute_chi1(traj)
    indices,angles2 = md.compute_chi2(traj)
    indices,angles3 = md.compute_chi3(traj)
    #indices,angles4 = md.compute_chi4(traj)
    #print(angles1)
    #print(type(angles1))
    #print(len(angles1))
    angles = np.concatenate((angles1,angles2,angles3), axis=1)
    dataset.append(angles)
    print("Done constructing dataset using chi angles")
    return dataset

def dataset_contacts(traj):
    dataset = []
    distances,residue_pairs = md.compute_contacts(traj, scheme='closest-heavy',
                                                  ignore_nonprotein=False, periodic=False, soft_min=False,
                                                  soft_min_beta=20)
    print("Done constructing dataset using residue contact information")
    dataset.append(distances)
    return dataset

def clustering(N_cluster_opt,dataset,traj):
    cluster = KMeans(n_clusters=N_cluster_opt, init='k-means++', n_init=10, max_iter=300, tol=0.001,
                     precompute_distances='auto',
                     verbose=0, random_state=None, copy_x=True, n_jobs=2).fit(dataset)
    cluster_centers = cluster.cluster_centers_
    print("center lenghth: " + str(len(cluster_centers)) + "\n")
    clusters = [[] for i in range(0, N_cluster_opt)]
    clusters_xyz = [[] for i in range(0, N_cluster_opt)]
    clusters_xyz_center = []
    fileout_labels=open("./AlleyCat-Ca-unconstrained/Labels_for_"+str(N_cluster_opt)+"_clusters.dat",'w')
    for i in range(0, len(cluster.labels_[0])):
        fileout_labels.write("snapshot "+str(i+1)+" corresponds to Cluster "+str(cluster.labels_[0][i]+1)+"\n")
        for j in range(0, N_cluster_opt):
            if cluster.labels_[0][i] == j:
                clusters[j].append(dataset[0][i])
                clusters_xyz[j].append(traj[i].xyz)
    fileout=open("./AlleyCat-Ca-unconstrained/population_for_"+str(N_cluster_opt)+"_clusters.dat",'w')
    for l in range(0, N_cluster_opt):
        clusters_xyz_center.append(np.average(np.array(clusters_xyz[l]), axis=0)[0])
        fileout.write('The population of cluster ' + str(l) + ' is ' + str(len(clusters[l]))+'\n')
        print('The population of cluster ' + str(l) + ' is ' + str(len(clusters[l])))
    fileout_labels.close()
    fileout.close()
    return clusters_xyz,clusters_xyz_center,cluster_centers,clusters, cluster.labels_[0]

def rePDB(N_cluster_opt):
    file1 = open("./AlleyCat-Ca-unconstrained/population_for_"+str(N_cluster_opt)+"_clusters.dat", 'r')
    file1_lines = file1.readlines()
    population = np.array([int(file1_lines[i].split()[6]) for i in range(len(file1_lines))])
    index = np.argsort(population)[::-1]
    file2 = open("./AlleyCat-Ca-unconstrained/cluster_center.pdb",'r')
    file2_lines = file2.readlines()
    length=0
    for k in range(len(file2_lines)):
        if 'ENDMDL' in file2_lines[k]:
            print("Frame length is "+str(k+1))
            length = k+1
            break
    file1_re = open("./AlleyCat-Ca-unconstrained/population_sorted.dat",'w')
    file2_re = open("./AlleyCat-Ca-unconstrained/cluster_center_sorted.pdb",'w')
    for i in index:
        file1_re.write(file1_lines[i])
        for j in range(length):
            file2_re.write(file2_lines[i*length+j])
    file1_re.close()
    file2_re.close()
    return 1

def main():
    Max_clusters=19
    Traj_interval=5
    traj_origin = md.load_netcdf('./AlleyCat-Ca-unconstrained/constprun3.mdcrd',top='./AlleyCat-Ca-unconstrained/AlleyCat_model1.prmtop')
    traj1=traj_origin[::Traj_interval]
    atomid = traj1.topology.select('resid 1 to 92')
    #atomid = traj1.topology.select("(resid 1 to 789 and backbone) or (resid 0)")
    #atomid = traj1.topology.select("(resid 0 152 160 277 278 326 334 339 340 434 436 450 643 645 765)")
    traj_pre = traj1.atom_slice(atomid)
    traj = traj_pre.superpose(traj_pre[0])
    traj_topo = traj1.topology.subset(atomid)
    del traj_origin, traj1, traj_pre
# dataset can be built by using different types of matrics. Here we used distance
    #dataset=dataset_CA_distances(traj)
    dataset = dataset_phi_psi_omega(traj)
    scale1 = StandardScaler(copy=True, with_mean=True, with_std=True)
    dataset_std = scale1.fit_transform(dataset[0])
    #dataset = dataset_phi(traj)
    # score functions loop over different number of Kmeans and then print corresponding inertia
    scores_in, scores_sc, scores_ch, scores_rt, scores_db = Kmeans_score([dataset_std], Max_clusters)
    #print(scores)
    #FST = np.gradient(scores)
    # Start clustering: Kmeans. n_jobs could be changed to allow parallel computing.
    Plot_scores(Max_clusters,scores_in,"inertia")
    Plot_scores(Max_clusters,scores_sc,"silhouette_coef")
    Plot_scores(Max_clusters,scores_ch,"calinski_harabasz")
    Plot_scores(Max_clusters,scores_rt, "ssr_sst_ratio")
    Plot_scores(Max_clusters, scores_db, "Davies-Bouldin Index")
    print("Done Kmean number analysis")
# Based on the above graph, you will find the optimal number of clusters.
# Clustering and collecting typical geometries
    N_cluster_opt = 6
# Define the number of clusters whose indexes will be printed.
    N_return_clusters = 20
    clusters_xyz, clusters_xyz_center, cluster_centers, clusters, labels = clustering(N_cluster_opt,[dataset_std],traj)
    avg_traj = md.Trajectory(np.array(clusters_xyz_center),traj_topo)
    avg_traj.save_pdb("./AlleyCat-Ca-unconstrained/cluster_center.pdb")
    dataset_center = dataset_phi_psi_omega(avg_traj)
    avg_traj.save_pdb("./cluster_center.pdb")
    dataset_center = dataset_phi_psi_omega(avg_traj)
    scale2 = StandardScaler(copy=True, with_mean=True, with_std=True)
    scale2.scale_ = scale1.scale_
    scale2.mean_ = scale1.mean_
    scale2.var_ = scale1.var_
    dataset_center_std = scale2.transform(dataset_center[0])
    pca1 = PCA(n_components=2)
    principalComponents = pca1.fit_transform(dataset_std)
    #cluster_center_std = StandardScaler().fit_transform(cluster_centers)
    projection_centers = np.matmul(
        np.array(cluster_centers).flatten().reshape(len(cluster_centers),-1),
        np.transpose(np.array(pca1.components_)))
    print(projection_centers)
    projection_ave = np.matmul(
        np.array(dataset_center_std).flatten().reshape(len(avg_traj),-1),
        np.transpose(np.array(pca1.components_)))
    projection_allpoints = []
    for i in range(0, N_cluster_opt):
        print("working on cluster: " + str(i)+"\n")
        projection_allpoints.append(
            np.matmul(np.array(clusters[i]).flatten().reshape(len(clusters[i]), -1),
                      np.transpose(np.array(pca1.components_))))
    #projection_allpoints[i][:, 0] projection_centers[:, 0]
    #projection_allpoints[i][:, 1] projection_centers[:, 1]
    Label_minidx=[]
    for i in range(0,N_cluster_opt):
        Distance_square=pow((projection_allpoints[i][:, 0]-projection_centers[i][0]),2)+pow((projection_allpoints[i][:, 1]-projection_centers[i][1]),2)
        Distance=pow(Distance_square,0.5)
        Label_minidx.append(np.argsort(Distance)[0:N_return_clusters])
    file_clus = open("./AlleyCat-Ca-unconstrained/nearest_clusters.dat",'w')
    for i in range(0,N_cluster_opt):
        A=np.sort(Label_minidx[i])
        B=np.argsort(Label_minidx[i])
        for k in range(0, len(Label_minidx[i])):
            N_counter = 0
            for j in range(0,len(labels)):
                if labels[j] == i and N_counter == A[k]:
                    file_clus.write("Cluster "+str(i)+" has snapshot: "+str(j+1)+" that ranks "+str(B[k]+1)+" closest to the center\n")
                    break
                elif labels[j] == i and N_counter != A[k]: N_counter=N_counter+1
    file_clus.close()
    plt.figure()
    se = ['gray', 'darksalmon', 'tan', 'palegreen', 'deepskyblue', 'plum', 'lemonchiffon', 'thistle',
          'lightpink','green']
    for i in range(0, N_cluster_opt):
        plt.scatter(projection_allpoints[i][:, 0], projection_allpoints[i][:, 1], marker='s', c=se[i])
        #plt.scatter(projection_allpoints[i][Label_minidx[i], 0], projection_allpoints[i][Label_minidx[i], 1], marker='^', c='r')
    plt.scatter(projection_centers[:, 0], projection_centers[:, 1], marker='o', c='r')
    plt.scatter(projection_ave[:, 0], projection_ave[:, 1], marker='x', c='k')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Pairwise distance PCA: AlleyCat')
    # cbar = plt.colorbar()
    # cbar.set_label('Time [ps]')
    plt.savefig('./AlleyCat-Ca-unconstrained/PCA.pdf', dpi=300)
    del traj, avg_traj
    rePDB(N_cluster_opt)





#######
main()