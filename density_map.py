from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas
'''
show all 4d information x,y,z,density as point cloud.
'''
# points = pandas.read_csv('densitymap_20w.csv')
points = pandas.read_csv('./density_point/density_0.csv')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = points['x'].values
y = points['y'].values
z = points['z'].values
d = points['d'].values

# ax.scatter(x, y, z, c=d, cmap='YlOrRd', marker='o')
ax.scatter(x, y, z, c=d, cmap='Reds', marker='o')

plt.show()