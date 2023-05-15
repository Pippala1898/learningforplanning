#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:38:50 2023

@author: pippala
"""

# Validate estimated 3D
from data_loader import PBW_3D, PBW_3D_est

def test_SG(sg1, sg2):
    set_of_tuples_1 = set([tuple(lst) for lst in sg1])
    set_of_tuples_2 = set([tuple(lst) for lst in sg2])
    if set_of_tuples_1 != set_of_tuples_2:
        return False
    
    return True


test_data = PBW_3D(train_size=1.0, nb_samples=-1)
est_data = PBW_3D_est(train_size=1.0, nb_samples=-1)
validation = []
for js in est_data.scene_jsons:
    sg_test = test_data.json2im[js][0]
    sg_est = est_data.json2im[js][0]
    validation.append(test_SG(sg_test, sg_est))
    
accuracy = validation.count(True)/len(validation)  

