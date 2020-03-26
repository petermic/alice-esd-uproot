import uproot
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

branch_list = [
    b"Tracks.fP[5]", # ALICE track parameter vector [Y,Z,sin(phi),dZ/ds,Q/pt]
    b"Tracks.fAlpha", # Local TPC slant angle
    b"Tracks.fTrackTime[5]", # Track TOF at various parts of detector (last one is at end of track)
    b"Tracks.fTrackLength" # Track length
]

def run_analysis():
    histogram = None

    if len(sys.argv) != 2:
        print("Usage: python uproot_analysis.py <input file list>")
        sys.exit()
    
    with open(sys.argv[1],"r") as filelist:
        files = filelist.read().splitlines()
        for entrystart,entrystop,data in uproot.iterate(files,b"esdTree;1",branch_list,reportentries=True):
            print("Processing entries "+str(entrystart)+"-"+str(entrystop)+"...")

            trackMomenta = mapGetMomentum(data)
            totalMomenta = np.sqrt(trackMomenta[:,0]**2+trackMomenta[:,1]**2+trackMomenta[:,2]**2)
            beta = getBeta(data)
            gamma = 1./np.sqrt(1-beta**2)
            trackMasses = totalMomenta/(beta*gamma)

            counts, edges = np.histogram(trackMasses,bins=1000,range=[0,10])
            if histogram is None:
                histogram = counts,edges
            else:
                histogram = histogram[0]+counts,edges
            
    counts, edges = histogram

    plt.step(x=edges, y=np.append(counts, 0), where="post")
    plt.xlim(edges[0], edges[-1])
    plt.yscale('log')
    plt.ylim(0, counts.max() * 1.1) 
    plt.show()

def getBeta(entry):
    t = entry[b"Tracks.fTrackTime[5]"]
    d = entry[b"Tracks.fTrackLength"]
    betas = []
    for i_evt in range(len(d)):
        for i_track in range(len(d[i_evt,:])):
            if d[i_evt,i_track] > 0:
                if t[i_evt,i_track,4] > 0:
                    b = ((d[i_evt,i_track]/100.) / (t[i_evt,i_track,4]/(10**12))) / (3*10**8)
                    betas.append(b)
    return np.array(betas)


def mapGetMomentum(entry):
    Pvectors = entry[b"Tracks.fP[5]"]
    alphas = entry[b"Tracks.fAlpha"]
    d = entry[b"Tracks.fTrackLength"]
    t = entry[b"Tracks.fTrackTime[5]"]
    momenta = []
    for i_evt in range(len(alphas)):
        for i_track in range(len(alphas[i_evt,:])):
            if d[i_evt,i_track] > 0 and t[i_evt,i_track,4] > 0:
                momenta.append(getMomentum(Pvectors[i_evt,i_track,:],alphas[i_evt,i_track]))
    return np.array(momenta)

# Momentum conversion formula adapted from an old ALICE public ROOT cookbook
def getMomentum(Pvector,alpha):
    p = [0,0,0]
    p[0] = Pvector[4]
    p[1] = Pvector[2]
    p[2] = Pvector[3]

    pt=1./math.fabs(p[0])
    cs=math.cos(alpha)
    sn=math.sin(alpha)
    r=math.sqrt(1 - p[1]*p[1])
    p[0]=pt*(r*cs - p[1]*sn)
    p[1]=pt*(p[1]*cs + r*sn)
    p[2]=pt*p[2]
    return np.array(p)

if __name__ == '__main__':
    run_analysis()