from sklearn.decomposition import PCA
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.animation as animation
from sklearn.cluster import KMeans
from cluster import cluster

def update_center(num, dataLines, lines):
    c = get_cluster(result[num])
    lines.set_data(center[c,0],center[c,1])

def update_lines(num, dataLines, lines, txt):
    lines.set_data(dataLines[num-5:num,0],dataLines[num-5:num,1])
    txt.set_text(str(num)+": "+str(get_cluster(dataLines[num])))

def get_cluster(pt):
    l = 10000000000
    c_ind = 0
    for i, ct in enumerate(center):
        d = ct - pt
        if d[0]**2+d[1]**2 < l:
            c_ind = i
            l = d[0]**2+d[1]**2
    return c_ind

def get_stationary_state(result, th_ratio=0.6):
    init = result[0]
    s_state = []
    dist = []
    for pt in result[1:]:
        d = np.linalg.norm(pt-init)
        dist.append(d)
        init = pt
    th = np.average(dist)*th_ratio
    for i in range(1,len(dist)):
        if dist[i-1] < th and dist[i] < th:
            s_state.append(result[i])
    return np.array(s_state)


data = np.load('jumpingjack_1_angle.npy')
pca = PCA(n_components=2)
result = pca.fit_transform(data)

np.save('jj_pca.npy',np.array(pca.components_))

result = np.matmul(data,pca.components_.T)
stationary = get_stationary_state(result)
print(stationary.shape)
kmeans = KMeans(n_clusters=3, random_state=0).fit(stationary)
center = kmeans.cluster_centers_
tr = kmeans.transform(stationary)
cluster_pt = [[] for _ in range(len(center))]

for pt, d in zip(stationary, tr):
    ind = np.argmin(d)
    cluster_pt[ind].append(pt)

c1 = cluster(np.array(cluster_pt[0]).T)
c1.approximate(6)

c2 = cluster(np.array(cluster_pt[1]).T)
c2.approximate(6)

c3 = cluster(np.array(cluster_pt[2]).T)
c3.approximate(6)

np.save('unit.npy',c1._unit_vector[2])
np.save('cluster/1.npy',np.array(c1.range).T)
np.save('cluster/2.npy',np.array(c2.range).T)
np.save('cluster/3.npy',np.array(c3.range).T)