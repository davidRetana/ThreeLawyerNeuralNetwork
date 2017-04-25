# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 14:47:33 2017

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt

def image_predictor(X, method):
    m, n = X.shape
    aux_n = np.sqrt(n)
    while True:
        option = input("Press Enter (q to exit): ")
        if option == 'q':
            break
        img = X[np.random.randint(0, m)]
        plt.imshow(img.reshape((aux_n, aux_n)), cmap='gray')
        plt.show()
        print("Label: ", method.predict(img))
        print("####################")
    return 0
