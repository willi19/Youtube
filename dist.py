from sklearn.decomposition import PCA
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.animation as animation

data = np.load('jumpingjack_1_angle.npy')
pca = PCA(n_components=2)
_ = pca.fit_transform(data)

np.save('jj_pca',pca.components_)

result = np.matmul(data,pca.components_.T)

init = result[0]
dist = []
for pt in result[1:]:
    d = np.linalg.norm(init-pt)
    dist.append(d)
    init = pt

print(np.average(dist))
plt.hist(dist,bins=1000)
plt.show()    


