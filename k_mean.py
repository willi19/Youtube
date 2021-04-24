from sklearn.decomposition import PCA
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.animation as animation
from sklearn.cluster import KMeans

data = np.load('jumpingjack_1_angle.npy')
pca = PCA(n_components=2)
_ = pca.fit_transform(data)

np.save('jj_pca',pca.components_)

result = np.matmul(data,pca.components_.T)
kmeans = KMeans(n_clusters=7, random_state=0).fit(result)

fig = plt.figure()
ax = plt.axes(xlim=(-10,10),ylim=(-10,10))

traj = ax.plot(result[:,0], result[:,1], 'ro',ms=1)[0]
center = ax.plot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],'bx',ms=10)[0]

ax.set_title('2D Test')

plt.show()

np.save('jj_center',kmeans.cluster_centers_)