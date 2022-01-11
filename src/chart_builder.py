import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def prices_to_pic():
    data = np.load("./data/input.npy")
    length = data.shape[0]
    images = np.zeros((len(data), 24, 32))
    #images = []
    for idx, i in enumerate(data):
        os.system("rm -rf plot.jpg")
        plt.plot(i, linewidth=30)
        plt.axis('off')
        plt.tight_layout(pad=0.05)
        #plt.show()
        plt.savefig("plot.jpg")
        plt.clf()
        im = Image.open("plot.jpg").convert('L')
        im = im.resize((32, 24))
        arr = np.asarray(im)
        images[idx,:,:] = arr
        print("%s/%s Done" % (idx, length))
    os.system("rm -rf plot.jpg")
    np.save("./data/input_images.npy", images)

prices_to_pic()
