from random import sample
from typing import final
import numpy as np
import matplotlib.pyplot as plt
import random



def generateNumbersF0(number_samples, desired_mean, desired_std_dev):
    # X1 sample 
    samples = np.random.normal(loc=0.0, scale=desired_std_dev, size=number_samples)

    actual_mean = np.mean(samples) 

    zero_mean_samples = samples - (actual_mean)
    zero_mean_std = np.std(zero_mean_samples)

    scaled_samples = zero_mean_samples * (desired_std_dev/zero_mean_std)

    final_samples = scaled_samples + desired_mean
    
    return final_samples


def generateNumbersF1(numbers_samples, desired_std_dev):
    binary_list = []
    
    samples = np.random.normal(loc=0, size=numbers_samples, scale=desired_std_dev)
    
    for i in range(numbers_samples):
        random_number = random.randint(0,1)
        if (random_number < 0.5):
            binary_list.append(-1)
        else: 
            binary_list.append(1)
    
    samples = samples + binary_list
    samples_std = np.std(samples)
    
    final_samples = samples * (desired_std_dev/samples_std)
    
    return final_samples

        
        


def createListPairs(array_one, array_two):
    final_list = []
    if (len(array_one) != len(array_two)):
        return

    for i in range(len(array_one)):
        number_pair = [array_one[i], array_two[i]]
        final_list.append(number_pair)
        
    return final_list
    
    
    


if __name__ == '__main__':
    num_samples = 1000000
    desired_mean = 0.0
    desired_std_dev = 1

    final_samples_1_F0 = generateNumbersF0(number_samples=num_samples, desired_mean=desired_mean, desired_std_dev=desired_std_dev)
    final_samples_2_F0 = generateNumbersF0(number_samples=num_samples, desired_mean=desired_mean, desired_std_dev=desired_std_dev)

    # Final pair data f0
    final_list_F0 = createListPairs(final_samples_1_F0, final_samples_2_F0)
    
    final_samples_1_F1 = generateNumbersF1(numbers_samples=num_samples, desired_std_dev=desired_std_dev)
    final_samples_2_F1 = generateNumbersF1(numbers_samples=num_samples, desired_std_dev=desired_std_dev)
    
    # Final pair data f1
    final_list_F1 = createListPairs(final_samples_1_F1, final_samples_2_F1)

    
    
    