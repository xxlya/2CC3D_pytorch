import deepdish as dd
import multiprocessing
from nilearn import image
import nibabel as nib
import os
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from scipy.io import loadmat
import numpy as np
import pandas as pd
import h5py

sub_dir = '/basket/Biopoint_3DConv_Classification_pytorch/MAT/subject.mat'
subjects = loadmat(sub_dir)
subjects_list = list(subjects['con'][0])+list(subjects['pat'][0])
data_dir = '/data3/sdb/Data/Biopoint/Freesurfer'
save_dir = '/basket/Biopoint/Data/data/Biopoint/h5_2c_win9'
csv_dir = '/basket/Biopoint_3DConv_Classification_pytorch/MAT/biopoint_subs.csv'
csv = pd.read_csv(csv_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def save_h5(sub_id):
    filename = 'tfMRI/res4d-reorient-brain-mni152_2mm_fnirt.nii.gz'
    h5_name =  str(sub_id) + '.h5'
    hf = nib.load(os.path.join(data_dir, str(sub_id), filename))
    fMRI = np.array(hf.dataobj)
    dd.io.save(os.path.join(save_dir,h5_name),{'fMRI':fMRI})

cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)

def save_h5_down(sub_id):
    filename = 'tfMRI/res4d-reorient-brain-mni152_2mm_fnirt.nii.gz'
    h5_name =  str(sub_id) + '.h5'
    hf = nib.load(os.path.join(data_dir, str(sub_id), filename))
    fMRI = np.array(hf.dataobj)
    shape = [91,109,91]
    steps = [0.5, 0.5, 0.5]  # original step sizes
    x, y, z = [steps[k] * np.arange(shape[k]) for k in range(3)]  # original grid
    new_fMRI = np.zeros((45,54,45,146))
    for t in range(146):
        f = RegularGridInterpolator((x, y, z), fMRI[:,:,:,t])  # interpolator
        dx, dy, dz = 1.0, 1.0, 1.0  # new step sizes
        new_grid = np.mgrid[0:x[-1]:dx, 0:y[-1]:dy, 0:z[-1]:dz]  # new grid
        new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2))  # reorder axes for evaluation
        new_values = f(new_grid)
        new_fMRI[:,:,:,t] = new_values
    dd.io.save(os.path.join(save_dir,h5_name),{'fMRI':new_fMRI})

def moving(a, n):
    res_std = np.zeros((a.shape[0],a.shape[1],a.shape[2],a.shape[3]-n+1))
    res_avg = np.zeros((a.shape[0],a.shape[1],a.shape[2],a.shape[3]-n+1))
    for i in range(0,a.shape[-1]-n+1):
        res_avg[:,:,:,i] = np.mean(a[:,:,:,i:i+n],axis=-1)
        res_std[:,:,:,i] = np.std(a[:,:,:,i:i+n],axis=-1)
    return res_avg, res_std

def save_h5_2c(sub_id):
    data_dir = '/basket/Biopoint/Data/data/Biopoint'
    filename = str(sub_id) + '.h5'
    h5_name = str(sub_id) #+ '.h5'
    hf = h5py.File(os.path.join(data_dir, 'h5_downsample',filename), 'r')
    #hf = dd.io.load(os.path.join(data_dir, 'h5_downsample',filename))
    res_avg, res_std = moving(hf['fMRI'].value,n=9)
    hf.close()
    res_std = np.nan_to_num(res_std)
    res_avg = np.nan_to_num(res_avg)
    label = csv[csv['SUB_ID']==sub_id]['DX_GROUP'].values[0]%2
    for i in range(res_avg.shape[-1]):
        dd.io.save(os.path.join(save_dir,h5_name+'_'+str(i)+'.h5'),{'avg':res_avg[:,:,:,i], 'std':res_std[:,:,:,i], 'label': label, 'id': sub_id})



cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)

#save_h5_2c(csv['SUB_ID'].values[0])


import timeit

start = timeit.default_timer()

pool.map(save_h5_2c,subjects_list)

stop = timeit.default_timer()

print('Time: ', stop - start)