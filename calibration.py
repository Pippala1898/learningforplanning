#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:25:39 2023

@author: pippala
"""

from data_loader import  PBW
from models import LocationBasedGeneratorCoordConv, LocationBasedGenerator
import torch
import argparse
import pickle

data_dir = "/Users/pippala/Desktop/robotic/location-based-generative-master/photorealistic-blocksworld-master/blocks-5-3"
device  = 'cpu'



def im_names(PBW_data):
    """
    Given an instance of PBW --> a dictionary with key = the order of the image in file ; value: masks and coordinates
    """
    file2coordnmask = dict()
    for file_name in PBW_data.keys:
        file2coordnmask[file_name] = [PBW_data.data[file_name][0],  PBW_data.data[file_name][-1]]
    return file2coordnmask


def scenario_cord( file_name, file2coordnmask, num_cameras = 2):
    '''
    if num_cameras = 1, return a list --> [predicated _2D_coord of objects, true_3D_coord of objects]
    
    if num_cameras = 2, return a list --> [predicated _2D_coord of objects for cam1 and cam2 , true_3D_coord of objects for cam1 and cam2 ]
    
    if num_cameras = 2, return a list --> [predicated _2D_coord of objects for all cams in order cam1, cam2, centercame, true_3D_coord of objects for all cams ]
 
    '''
    if file_name not in file2coordnmask:
        return None
    file_name_prefix = file_name[:-9]
    files = []
    pred_2D_coord = []
    all_real_coord = []
    if file_name[-9:] in ['cam1.json', 'cam2.json']:
        file_name_cam1=file_name_prefix + 'cam1.json'
        file_name_cam2=file_name_prefix + 'cam2.json'
        file_name_center=file_name_prefix[:-1]+'.json'
    else:
        file_name_cam1=file_name[:-5] + '_cam1.json'
        file_name_cam2=file_name[:-5] + '_cam2.json'
        file_name_center=file_name 
        
        
    
    if num_cameras ==1:
        files = [file_name_center]
    elif num_cameras ==2:
        files = [file_name_cam1, file_name_cam2]
    else:
        files = [file_name_cam1, file_name_cam2, file_name_center]
        
    for file_name in files:    
        masks = file2coordnmask[file_name][0]
        real_3D_coord = torch.tensor([file2coordnmask[file_name][1]]) 
        masks = torch.tensor(masks).to(device)
        masks = masks.view(-1, 3, 128, 128)
        theta = model.find_theta(masks).view(1, -1, 6).detach().numpy()
        coordinates = theta[:, :, [2, 5]]
        coordinates[:, :, 0] *= -1    
        coordinates = torch.tensor(coordinates)
        pred_2D_coord.append(coordinates)
        all_real_coord.append(real_3D_coord)
    out = (torch.cat(pred_2D_coord, dim=0), torch.cat(all_real_coord, dim=0))  
    return out 







if __name__ == '__main__':
    PBW_all = PBW(root_dir=data_dir, train_size=1, nb_samples=-1) ## PBW_all[0] ---> (all_inp, targets, original_inp, weight_maps, sg, obj_names)
    model = LocationBasedGenerator()
    model.to(device)
    model.load_state_dict(torch.load("pre_models/model-20230424-030827", map_location=device))
    file2coordnmask = im_names(PBW_all)    
    parser = argparse.ArgumentParser(description='2D to 3D convert.')
    parser.add_argument('--filename', type=str, help='A .json file of the scenario', default = '/Users/pippala/Desktop/robotic/location-based-generative-master/photorealistic-blocksworld-master/blocks-5-3/scene/CLEVR_new_000804_cam1.json')
    parser.add_argument('--cameras', type=int, help='number of cameras', default = 2)

    args = parser.parse_args()    

    image_coords , real_coords = scenario_cord( args.filename, file2coordnmask, args.cameras)


    





    
## Save PBW_all   
# import pickle
# PBW_all = PBW(root_dir=data_dir, train_size=1, nb_samples=-1) 
# with open('block5_3.pkl', 'wb') as f:
#     pickle.dump(PBW_all, f)
# # Load the dataloader
# with open('block5_3.pkl', 'rb') as f:
#     PBW_all = pickle.load(f)
    


## objects coordinates
# my_list = []
# for num in range(1953):
#     my_list.append('{:06}'.format(num))


# dictionary = dict()    
# for file in my_list:
#    file_name = '/Users/pippala/Desktop/robotic/location-based-generative-master/photorealistic-blocksworld-master/blocks-5-3/scene/CLEVR_new_'+file+'.json'    
#    dictionary[file_name] =  scenario_cord( file_name, file2coordnmask, 3) 

# with open('coordinates.pickle', 'wb') as f:
#     pickle.dump(dictionary, f)

# with open('coordinates.pickle', 'rb') as f:
#     coordinates = pickle.load(f)


