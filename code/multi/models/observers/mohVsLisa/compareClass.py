import os,glob,cv2
import sys,argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# python compareClass.py ../../../processedDataset/datasetV1.0/lisaSet/images/

image_path=sys.argv[1]
classes = os.listdir(image_path)

moh0 = [0,0,0,0,
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

moh1 = [0,0,0,1,
        1,1,0,2,
        1,0,2,1,
        2,1,1,0,
        2,2,2,2,
        0,0,0,0,
        1]

moh2 = [2,2,1,1,
        1,2,2,1]
moh = moh0+moh1+moh2
lisa = []
x = []

summary = []

def plot_confusion_matrix(y_true, y_pred, classes, normalize,
                          title=None, cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	title = 'Confusion matrix'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print("Normalized confusion matrix")


	print(cm)

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		yticks=np.arange(cm.shape[0]),
		# ... and label them with the respective list entries
		xticklabels=classes, yticklabels=classes,
		title=title,
		ylabel='Lisa\'s True label',
		xlabel='Moh\'s predicted label')

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
i = 0
for fields in classes:   
    path = os.path.join(image_path, fields, '*g')
    files = glob.glob(path)
    for f in files:
        p = f.split('processedDataset/datasetV1.0/lisaSet/images/Phase')
        file = p[1].split('Copy of ')
        x.append(file[2])
        if "0" in fields:
            lisa.append(0)
        elif "1" in fields:
            lisa.append(1)
        else:
            lisa.append(2)
        summary.append("Image: "+p[1]+",    "+ str(moh[i])+",    "+ str(lisa[i]))
        print("Image:"+p[1]+","+ str(moh[i])+","+ str(lisa[i]))
        i += 1

class_names = ['0','1','2']
plot_confusion_matrix(lisa, moh, classes=class_names, 
						normalize=True, title='Confusion matrix')
plt.show()