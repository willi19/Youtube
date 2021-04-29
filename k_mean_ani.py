from sklearn.decomposition import PCA
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.animation as animation
from sklearn.cluster import KMeans

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

def get_stationary_state(result, th_ratio=0.5):
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

result = np.matmul(data,pca.components_.T)
stationary = get_stationary_state(result)
print(stationary.shape)
kmeans = KMeans(n_clusters=3, random_state=0).fit(stationary)
center = kmeans.cluster_centers_
tr = kmeans.transform(stationary)
dist = [[] for _ in range(len(center))]

for d in tr:
    ind = np.argmin(d)
    dist[ind].append(np.min(d))
    if np.min(d) > 2:
        continue

mean_rad = [np.average(dist[i]) for i in range(len(center))]
max_rad = [np.max(dist[i]) for i in range(len(center))]

fig, ax = plt.subplots()
ax.set_xlim((-10,10))
ax.set_ylim((-10,10))

print(mean_rad)
print(max_rad)

for i in range(len(center)):
    mean_circle = plt.Circle(center[i],mean_rad[i],fill=False)
    ax.add_artist(mean_circle)

for i in range(len(center)):
    max_circle = plt.Circle(center[i],max_rad[i],fill=False)
    ax.add_artist(max_circle)
    

#traj = ax.plot(result[:,0], result[:,1], 'ro',ms=1)[0]
line = ax.plot(result[0:1,0], result[0:1,1],'bo',ms=7)[0]
plot_center = ax.plot(center[:,0],center[:,1],'ro',ms=5)[0]
stationary = ax.plot(stationary[:,0],stationary[:,1],'go',ms=3)[0]
txt = ax.text(-9,-9,get_cluster(result[0]))
ax.set_title('2D Test')
line_ani = animation.FuncAnimation(fig, update_lines, len(result)-1, fargs=(result, line, txt),
                                   interval=100, blit=False)

plt.show()

np.save('jj_center',kmeans.cluster_centers_)