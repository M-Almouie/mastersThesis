import os,glob,cv2
import sys,argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# python compareClass.py ../../../processedDataset/datasetV1.0/lisaSet/images/

image_path=sys.argv[1]
classes = os.listdir(image_path)

blindOrNormal0 = [0,0,0,0,  0,0,0,0,  0,0,0,0,   0,0,0,0,   0,0,0,0,   0,0,0,0,   0,0,0,0,   0,0,0,0,   0,0,0,0,
        		  0,0,0,0,   0,0,0,0,  0,0,0,0,   0,0,0,0,   0,0,0]

blindOrNormal1 = [0,0,0,0,  1,1,0,1,  1,0,1,1,   1,1,1,0,   1,1,1,1,   1,0,0,0, 0]

blindOrNormal2 = [1,1,1,1, 1,1,1,1]

blindOrNormal = blindOrNormal0+blindOrNormal1+blindOrNormal2

mich0 = [1,0,0,0,   0,0,1,0,  0,0,0,0,   1,0,0,0,   0,0,0,0,   
		 0,0,0,0,   0,1,0,0,  0,0,0,0,   0,0,0,0,
         0,0,0,1,   1,0,0,0,  0,0,0,0,   0,1,0,0,   0,0,0]

moh0 =  [0,0,0,0,   0,0,0,0,  0,0,0,0,   0,0,0,0,   0,0,0,0,
	     0,0,0,0,   0,0,0,0,  0,0,0,0,   0,0,0,0,
         0,0,0,0,   0,0,0,0,  0,0,0,0,   0,0,0,0,   0,0,0]


mich1 = [1,0,0,2,  0,2,0,2,  2,0,2,2,   2,2,2,1,   2,2,2,2,   
		 1,1,1,0,  0]

moh1  = [0,0,0,1,  1,1,0,2,  1,0,2,1,   2,1,1,0,   2,2,2,2,
		 0,0,0,0,  1]


mich2 = [2,2,2,2,  1,2,2,2]

moh2  = [2,2,1,1,  1,2,2,1]


lisa0 = [0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0]

lisa1 = [1,1,1,1,
        1,1,1,1,
        1,1,1,1,
        1,1,1,1,
        1,1,1,1,
        1,1,1,1,
        1]

lisa2 = [2,2,2,2,
		 2,2,2,2]

mich = mich0+mich1+mich2
moh = moh0+moh1+moh2
lisa = lisa0+lisa1+lisa2

x = []

summary = []

def plot_confusion_matrix(y_true, y_pred, titleName, classes, normalize,
                          cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	title = titleName

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print("Normalized confusion matrix")


	print(cm)
	return
	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		yticks=np.arange(cm.shape[0]),
		# ... and label them with the respective list entries
		xticklabels=classes, yticklabels=classes,
		title=title,
		ylabel='True label',
		xlabel='Michael\'s predicted label')

    # Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
				ha="center", va="center",
				color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	return ax

class_names = ['0','1','2']
title = 'Blind vs Normal Confusion matrix'
#plot_confusion_matrix(lisa, moh, classes=class_names,normalize=True, title='Confusion matrix')
#plot_confusion_matrix(blindOrNormal,mich, title,classes=class_names, normalize=True)
plot_confusion_matrix(lisa, moh, title,classes=class_names, normalize=False)

#plt.show()

# This import registers the 3D projection, but is otherwise unused.


########
# 3d confusion histograms
# Fixing random state for reproducibility
np.random.seed(19680801)


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
x = moh
y = mich
hist, xedges, yedges = np.histogram2d(x, y, bins=3, range=[[0, 2], [0, 2]])

xedges = np.array([0,1,2])
yedges = xedges

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges, yedges, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5
dz = hist.ravel()
ax.set_title('Mohamed vs Micheal Confusion')
ax.set_xlabel('Mohamed Prediction labels')
ax.set_ylabel('Michael True labels')
ax.bar3d(xpos, ypos, zpos, dx, dy, dz)

plt.show()
