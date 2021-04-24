from sklearn.decomposition import PCA
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.animation as animation

data = np.load('jumpingjack_1_angle.npy')
pca = PCA(n_components=2)
result = pca.fit_transform(data)

result = result.T

def update_line(num, dataLines, lines):
    x,y,z = result[:,num]
    lines.set_data(np.array(x),np.array(z))
    lines.set_3d_properties(1-np.array(y))

def update_lines(num, dataLines, lines):
    lines.set_data(result[0:2,:num])

fig = plt.figure()
ax = plt.axes(xlim=(-7,7),ylim=(-7,7))
line = ax.plot([], [], 'o',ms=1)[0]


ax.set_title('2D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, 12057, fargs=(result, line),
                                   interval=50, blit=False)

plt.show()