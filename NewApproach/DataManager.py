import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import csv

def f0_samples(n_samples):
    return np.random.randn(n_samples)

def f1_samples(n_samples):
    x = f0_samples(n_samples)
    blist = [ 1.0 if np.random.uniform(0.0,1.0)<0.5 else -1.0 for _ in range(n_samples)]
    return x + blist

def write_to_csv(file_name, samples):
    with open(file_name, 'w') as file:
        write = csv.writer(file)
        write.writerow(['X', 'Y'])
        write.writerows(samples)
    
def create_samples(n_samples):
    f0_x1 = f0_samples(n_samples=n_samples)
    f0_x2 = f0_samples(n_samples=n_samples)

    f0 = np.array([[x1,x2] for x1,x2 in zip(f0_x1,f0_x2)])

    f1_x1 = f1_samples(n_samples=n_samples)
    f1_x2 = f1_samples(n_samples=n_samples)

    f1 = np.array([[x1,x2] for x1,x2 in zip(f1_x1,f1_x2)])

    return (f0,f1)

def save_samples(samples,title):
    index=0
    for sample in samples:
        write_to_csv('f' + str(index) + '_' + title + '.csv' , sample)
        index+=1

if __name__ == '__main__':
    n_samples = 1000000
    n_test = 200
    bins = 200

    f0,f1 = create_samples(n_samples)
    save_samples([f0,f1],"test_samples")
    fig, axes = plt.subplots(nrows=2, ncols=2)

    plt.subplot(2, 2, 1)
    plt.hist(f0[:,0], bins=bins)
    plt.subplot(2, 2, 2)
    plt.hist(f0[:,1], bins=bins)
    plt.subplot(2, 2, 3)
    plt.hist(f1[:,0], bins=bins)
    plt.subplot(2, 2, 4)
    plt.hist(f1[:,1], bins=bins)

    plt.tight_layout()
    plt.show()



    




