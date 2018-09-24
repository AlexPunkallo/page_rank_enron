
# Alessandro Gallo

import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
import time
import matplotlib.pyplot as plt

graph_query = sys.argv[1]
alpha = sys.argv[2]

enron_graph=open(graph_query, "r")
sub_element=[]
for line in enron_graph.readlines():
    line=line.split(",")
    sub_element.append((float(line[0]),float(line[1][:-1])))
dim=max(map(max,zip(*sub_element)))+1
compress=dok_matrix((dim,dim), dtype=float)
compress.update({item:1. for item in sub_element})
compress=compress.tocsr()
tot=np.array(compress.sum(axis=0))[0]
ind=compress.nonzero()[1]
compress.data/=tot[ind]

def PageRank(M, alpha=0.86, epsilon=10**(-10)):
    n=M.shape[0]
    old_page=np.array([1./n for item in range(n)])
    converged=False
    iter_num=0
    start = time.time()
    while not converged:
        new_page=alpha*(M.dot(old_page))-(1-alpha)/n
        new_page+=(1-sum(new_page))/n
        if (np.linalg.norm(new_page-old_page)<=epsilon):
            converged=True
        iter_num+=1
        old_page=new_page
    end = time.time()
    print "\nn= %s, M.nnz= %s, density= %s" % (int(dim), compress.nnz, compress.nnz/(dim**2))
    print "iterations= %s, elapsed= %s " % (iter_num, (end - start))  
    return new_page

p=PageRank(compress)
print ("min(p), max(p), avg(p), sum(p) = %s, %s, %s, %s" % (np.min(p), np.max(p), np.mean(p), sum(p)))
print "\n	 Loading plot...\n"

points_numb=np.arange(min(p),max(p), 10**(-5))
values=[sum([1 if item>prob else 0 for item in p]) for prob in points_numb]
plt.xscale("log")
plt.yscale("log")
plt.scatter(points_numb,values)
plt.show()

