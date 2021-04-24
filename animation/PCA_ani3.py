from sklearn.decomposition import PCA
import numpy as np 
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

data = np.load('jumpingjack_1_angle.npy')
print(data.shape)
pca = PCA(n_components=3)
result = pca.fit_transform(data)

print(pca.explained_variance_ratio_)
print(result.shape)

result = result.T

# Fixing random state for reproducibility
np.random.seed(19680801)


def update_line(num, dataLines, lines):
    x,y,z = result[:,num]
    lines.set_data(np.array(x),np.array(z))
    lines.set_3d_properties(1-np.array(y))

def update_lines(num, dataLines, lines):
    lines.set_data(result[0:2,num-10:num])
    lines.set_3d_properties(result[2, num-10:num])

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Fifty lines of random 3-D lines
#data = [Gen_RandLine(500, 3) for index in range(5)]

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
x,y,z = result[:,0]
lines = ax.plot(result[0,0:1], result[2,0:1], result[1,0:1],'o',ms=1)[0]

# Setting the axes properties
ax.set_xlim3d([-10, 10])
ax.set_xlabel('X')

ax.set_ylim3d([-10, 10])
ax.set_ylabel('Y')

ax.set_zlim3d([-10, 10])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, 12057, fargs=(result, lines),
                                   interval=50, blit=False)

plt.show()