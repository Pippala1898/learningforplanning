#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 22:31:53 2023

@author: pippala
"""

from data_loader import PBW_3D,  PBW_3D_est
from models import LocationBasedGenerator
import queue
import torch
import sys
import random
import torchvision
import pickle
import utils
from utils import show2
from PIL import Image
from queue import Queue


def find_top(up_rel, ob_start):
    '''
    Find the top of ob_start and the corresponding relation
    '''
    rel_res = None
    while True:
        done = True
        for rel in up_rel:
            if rel[2] == ob_start:
                ob_start = rel[0]
                done = False
                rel_res = rel
                break
        if done:
            return ob_start, rel_res
        


def visualize_plan(im_list, perrow=9, if_save=False, name="solution"):
    ''' Visualize our action plan'''
    im_tensors = []
    transform = torchvision.transforms.ToTensor()
    for idx, im_name in enumerate(im_list): 
        im = Image.open(im_name).convert('RGB')
        im_tensors.append(transform(im).unsqueeze(0))
        if if_save:
            im.save("figures/%d.png" % idx)
    im_tensors = torch.cat(im_tensors, dim=0)
    show2([im_tensors], name, perrow)

        

def find_closest(up_rel, ob_start):
    '''
    find the object up to the base
    '''
    for rel in up_rel:
        if rel[2] == ob_start:
            return rel[0]

        


def obs_rel(sg, ob_names):
    '''
    find the relationships of each objects
    each object has only one up 
    '''
    relations = [ele.copy() for ele in sg]
    relationships = [rel.copy() for rel in sg]
    obj_names = ob_names + ["00", "01", "02", "10", "11", "12"]
    my_dict = {key: None for key in obj_names}
    
    for ob in obj_names:  
       # ob_rel = [] 
       if not relationships:
           break
       for rel in relationships:
           if rel[2] == ob and rel[1] == 'up':
               if my_dict[ob] is not None:
                   raise Error
                   
               my_dict[ob] = rel
               relations.remove(rel)
       relationships = [ele.copy() for ele in relations]   
    return my_dict           
               
 
        



    
def sg_cost(sg, ob_names, sg_to_relation):
    '''
    num of objects at wrong location 
    ob_name: as set
    '''
    # checking from base to top
    same_layer_ob = ["00", "01", "02", "10", "11", "12"]
    correct_ob = []    
    objects = set(ob_names)
    sg_up = [rel.copy() for rel in sg if rel[1] == "up"]
    sg_relation = obs_rel(sg, ob_names)
    # sg_to_relation = obs_rel(sg_to, ob_names)
    
    while same_layer_ob:
        correct_ob = []        
        for ob in same_layer_ob: 
            objects.discard(ob)              
            rel = sg_to_relation[ob] # relation of the object ob           
            if sg_relation[ob] == rel and rel is not None:                                
                correct_ob.append(rel[0])
                
        same_layer_ob = correct_ob[:]   
    return len(objects)         
        





def action_model(sg_from):
    '''the next scene graph after an action'''
    all_bases = ["00", "01", "02", "10", "11", "12"]
    present_bases = [rel[2] for rel in sg_from if rel[2] in all_bases]
    up_rel = [rel for rel in sg_from if rel[1] == "up"]
    neighboring_sg = []

    for b1 in present_bases[:]:
        for b2 in all_bases[:]:
            if b1 != b2:
                o1, r1 = find_top(up_rel, b1)
                o2, r2 = find_top(up_rel, b2)

                sg_neighbor = sg_from[:]
                sg_neighbor.remove(r1)
                for rel in sg_neighbor:
                    # all relation with o1
                    if rel[0] == o1 or rel[2] == o1:
                        sg_neighbor.remove(rel)
                # new relation 
                sg_neighbor.append([o1, "up", o2])
                if r2 is None:
                    if b2[0] == "0":
                        behind_base = "1"+b2[1] # the behind location                            
                        if behind_base in present_bases:                            
                            behind_obj = find_closest(up_rel, behind_base) # the behind object
                            if o1 != behind_obj:
                                sg_neighbor.append([o1, "front", behind_obj])
                    elif b2[0] == "1":
                        front_base = "0" + b2[1]
                        if front_base in present_bases:
                            front_obj = find_closest(up_rel, front_base)
                            if o1 != front_obj:
                                sg_neighbor.append([front_obj, "front", o1])
                neighboring_sg.append([sg_neighbor, "%s->%s" % (b1, b2)])
    return neighboring_sg

       




def hash_sg(relationships,
            ob_names=('brown', 'purple', 'cyan', 'blue', 'red', 'green', 'gray','turquoise',
                      "00", "01", "02", "10", "11", "12")):
    """
    hash into unique ID
    :param relationships: [['brown', 'left', 'purple'] , ['yellow', 'up', 'yellow']]
    :param ob_names:
    :return:
    """
    for rel in relationships:
        assert rel[0] in ob_names and rel[2] in ob_names  

    a_key = [0] * len(ob_names) * len(ob_names)
    pred2id = {"none": 0, "left": 1, "up": 2, "front": 3}
    predefined_objects1 = ob_names[:]
    predefined_objects2 = ob_names[:]
    pair2pred = {}
    for rel in relationships:
        if rel[1] != "__in_image__":
            pair2pred[(rel[0], rel[2])] = pred2id[rel[1]]

    idx = 0
    for ob1 in predefined_objects1:
        for ob2 in predefined_objects2:
            if (ob1, ob2) in pair2pred:
                a_key[idx] = pair2pred[(ob1, ob2)]
            idx += 1
    return tuple(a_key)



def a_star_search(sg_from, sg_to, hash_func, sg_cost):
    '''A* search'''
    bases = ['00', '01', '02', '10', '11', '12']
    ob_names = ({rel[0] for rel in sg_to }).union({rel[2] for rel in sg_to})
    ob_names = [ ele for ele in ob_names if ele not in bases]
    
    up_rel = [rel for rel in sg_to if rel[1] == "up"]

    sg_to_relation = obs_rel(sg_to, ob_names)
    
    open_set = [sg_from]
    path = {}

    g_score = {hash_func(sg_from): 0}
    f_score = {hash_func(sg_from): sg_cost(sg_from, ob_names,  sg_to_relation)}
    
    

    while len(open_set) > 0:
        current = min(open_set, key=lambda sg: f_score[hash_func(sg)])
        if hash_func(current) == hash_func(sg_to):
            return hash_func(current), path
        open_set.remove(current)

        for neighbor, act in action_model(current):
            tentative_g_score = g_score[hash_func(current)]+1
            if hash_func(neighbor) not in g_score or tentative_g_score < g_score[hash_func(neighbor)]:
                path[hash_func(neighbor)] = (hash_func(current), act)
                g_score[hash_func(neighbor)] = tentative_g_score
                f_score[hash_func(neighbor)] = g_score[hash_func(neighbor)] + sg_cost(neighbor, ob_names, sg_to_relation)
                if neighbor not in open_set:
                    open_set.append(neighbor)
    return None




def recon_path(path, hash_sg_to, hash_sg_from):
    '''reorder the actions'''
    act = []
    current = hash_sg_to
    res = []

    while current != hash_sg_from:
        res.append(current)
        current , a = path[current]
        act.append(a)
    res.append(hash_sg_from)
    return res, act


        

def plan(nb_plans=3):
    '''randomly pick some scenarios for testing'''
    root_dir = "/Users/pippala/Desktop/robotic/location-based-generative-master/photorealistic-blocksworld-master/blocks-5-3/image"
    test_data = PBW_3D(train_size=1.0, nb_samples=-1)
    est_data = PBW_3D_est(train_size=1.0, nb_samples=-1)
    sg2behind = test_data.json2im["sg2behind"]
    
    images_dir = []
    for n in range(nb_plans):
        # sg_from = random.choice(list(test_data.json2im.values()))[0] # scene graph
        # sg_to = random.choice(list(test_data.json2im.values()))[0]
        sg_from = random.choice(list(est_data.json2im.values()))[0] # scene graph
        sg_to = random.choice(list(est_data.json2im.values()))[0]        
        print(sg2behind[est_data.hash_sg(sg_from)][-1], "to",
              sg2behind[est_data.hash_sg(sg_to)][-1])
        goal , path  = a_star_search(sg_from, sg_to, hash_sg, sg_cost)
        # goal, path = graph_search(sg_from, sg_to, test_data.hash_sg)
        ## The action is goal to initial 
        state_path, actions = recon_path(path, goal,
                                          test_data.hash_sg(sg_from))
        # if len(actions) > 3:
        im_list = []
        state_path.reverse()
        # JUST For DEMO
        for state in state_path:
            if nb_plans ==1:
               images_dir.append(sg2behind[state][-1]) 
            print(sg2behind[state][-1])
            im_list.append("%s/%s" % (root_dir, sg2behind[state][-1]))
        visualize_plan(im_list, name="solution%d" % n)

    return  images_dir       
    

    
    
if __name__ == '__main__':
    plan(nb_plans=1)




## Some outcomes 
# CLEVR_new_000671.png to CLEVR_new_001174.png
# CLEVR_new_000671.png
# CLEVR_new_001343.png
# CLEVR_new_001175.png
# CLEVR_new_001139.png
# CLEVR_new_001138.png
# CLEVR_new_001174.png
# saved to solution0

# CLEVR_new_001689.png to CLEVR_new_000555.png
# CLEVR_new_001689.png
# CLEVR_new_001683.png
# CLEVR_new_000339.png
# CLEVR_new_000345.png
# CLEVR_new_000555.png
# saved to solution0
# CLEVR_new_000915.png to CLEVR_new_001341.png
# CLEVR_new_000915.png
# CLEVR_new_000916.png
# CLEVR_new_001252.png
# CLEVR_new_001258.png
# CLEVR_new_001341.png
# saved to solution1
# CLEVR_new_000226.png to CLEVR_new_001255.png


# CLEVR_new_001517.png to CLEVR_new_000867.png
# CLEVR_new_001517.png
# CLEVR_new_001515.png
# CLEVR_new_001599.png
# CLEVR_new_000927.png
# CLEVR_new_000843.png
# CLEVR_new_000867.png
# saved to solution2
    
    
 
    

## Testing part 
# use test_data.scene_jsons[0] for file name 
# m = test_data.json2im[test_data.scene_jsons[0]]
#     scene_graph = m[0]       
   

     
## CLEVR_new_000647.png to CLEVR_new_000829.png
# [['gray', 'up', 'green'],
#  ['turquoise', 'up', '00'],
#  ['red', 'up', '01'],
#  ['blue', 'up', '11'],
#  ['green', 'up', '12'],
#  ['red', 'front', 'blue']]
## to 

# [['turquoise', 'up', '00'],
#  ['red', 'up', '02'],
#  ['gray', 'up', '10'],
#  ['blue', 'up', '12'],
#  ['green', 'up', '01'],
#  ['turquoise', 'front', 'gray'],
#  ['red', 'front', 'blue']]


#['11->12', '12->01', '12->10', '01->02']
#['01->02', '12->10', '12->01', '11->12']



