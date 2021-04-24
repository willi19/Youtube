from sklearn.decomposition import PCA
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.animation as animation
from sklearn.cluster import KMeans

def update_center(num, dataLines, lines):
    c = get_cluster(result[num])
    lines.set_data(center[c,0],center[c,1])

def update_lines(num, dataLines, lines):
    print(np.arange(0,num/10,1/10)[-100:].shape)
    print(dataLines[num-100:num,0].shape)
    lines.set_data(dataLines[num-100:num,0],np.arange(0,num/10,1/10)[-100:])

def get_cluster(pt):
    l = 10000000000
    c_ind = 0
    for i, ct in enumerate(center):
        d = ct - pt
        if d[0]**2+d[1]**2 < l:
            c_ind = i
            l = d[0]**2+d[1]**2
    return c_ind

data = np.load('jumpingjack_1_angle.npy')
pca = PCA(n_components=2)
result = pca.fit_transform(data)

result = np.matmul(data,pca.components_.T)
print(result.shape)
kmeans = KMeans(n_clusters=3, random_state=0).fit(result)
center = kmeans.cluster_centers_

fig = plt.figure()
ax = plt.axes(xlim=(-10,10),ylim=(-1,15))

#traj = ax.plot(result[:,0], result[:,1], 'ro',ms=1)[0]
line = ax.plot(result[0:1,0], result[0:1,1],ms=7)[0]
plot_center = ax.plot(center[:,0],center[:,1],'ro',ms=5)[0]

ax.set_title('2D Test')
line_ani = animation.FuncAnimation(fig, update_lines, 12057, fargs=(result, line),
                                   interval=32, blit=False)

plt.show()

np.save('jj_center',kmeans.cluster_centers_)