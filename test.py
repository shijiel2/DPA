import torch
import numpy as np

cert = torch.load('/home/student.unimelb.edu.au/shijiel2/shijie/DPA/radii/cifar_nin_baseline_partitions_15.pth')
cert = cert.cpu().detach().numpy()

np.save(f"dpa_cpsa.npy", cert)