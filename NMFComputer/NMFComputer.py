'''
Created on Jul 8, 2018

@author: daniel
'''
import numpy as np
import math
class NMFComputer():
    block_size = 0
    num_hist_bins = 0
    num_components = 0
    
    def __init__(self, block_size = 400, num_hist_bins = 256, num_components = 5):
        self.setBlockSize(block_size)
        self.setNumHistBins(num_hist_bins)
        self.setNumComponents(num_components)

    
    def setBlockSize(self, block_size):
        if block_size > 0:
            self.block_size = block_size
        else:
            print("Error: please enter a valid column window size!")

    
    def setNumHistBins(self, num_hist_bins):
        if num_hist_bins > 0:
            self.num_hist_bins = num_hist_bins
        else:
            print("Error: please enter a valid histogram bin size!")
    
    def setNumComponents(self, num_components):
        if num_components > 0:
            self.num_components = num_components
        else:
            print("Error: please enter a valid number of components!")
            
    def computeHistograms(self, matrix):
        hist_image = []
        m = math.sqrt(self.block_size)
        cols = np.hsplit(matrix, m)
        for c in cols:
            blocks = np.vsplit(c, m)
            for b in blocks:
                hist, _ = np.histogram(b,bins=self.num_hist_bins)
                hist_image.append(hist)

        #print(len(hist_image))
        return np.array(hist_image).transpose()
    
    def computeNMF(self, V):
        pass
    
    def run(self, image):
        V = self.computeHistograms(image)
        W, H = self.computeNMF(V)
        return W, H