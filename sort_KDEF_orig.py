from imutils import paths
import random
import os
import shutil

CLASSES = {'AN': "anger", 'DI': "disgust", 'AF': "fear", 'HA': "happiness",
           'SA': "sadness", 'SU': "surprise", 'NE': "neutral"}

imagePaths = list(paths.list_images('KDEF_orig'))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    # Extract filename and label
    filename = imagePath.split(os.path.sep)[-1]
    label = CLASSES[filename[4:6]]

    # Construct output directory for class
    classDir = os.path.sep.join(['KDEF_sorted', label])
    if not os.path.exists(classDir):
        os.makedirs(classDir)

    # Construct image path in output directory and copy image in
    p = os.path.sep.join([classDir, filename])
    shutil.copy2(imagePath, p)
