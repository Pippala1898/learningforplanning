import json
import torch
import torchvision
import numpy as np
import pickle
import time
import random
import math
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from os import listdir
from os.path import isfile, join
from PIL import Image

# root_dir_PWD = "image/CLEVR_new_000008_cam1.png"

root_dir_PWD = "/Users/pippala/Desktop/robotic/location-based-generative-master/photorealistic-blocksworld-master/blocks-5-3"

root_dir_PWD3D = "/Users/pippala/Desktop/robotic/location-based-generative-master/photorealistic-blocksworld-master/blocks-5-3"

root_dir_PBWrandom_loc = "/Users/pippala/Desktop/robotic/location-based-generative-master/photorealistic-blocksworld-master/blocks-5-3"

root_dir_sim  = '/Users/pippala/Desktop/robotic/location-based-generative-master/5objs_seg'



class PBW(Dataset):
    "loading Photorealistic data without 3D information "
    def __init__(self, root_dir=root_dir_PWD,
                 train=True, train_size=0.6, nb_samples=1000, json2im=None, if_save_data=True):
        print("Loading from", root_dir)
        super(PBW, self).__init__()
        self.root_dir = root_dir
        self.train = train
        identifier = root_dir.split("/")[-1]

        json_dir = "%s/scene" % root_dir
        self.scene_jsons = [join(json_dir, f) for f in listdir(json_dir) if isfile(join(json_dir, f))] # alist of file name 

        self.image_dir = "%s/image" % root_dir
        self.transform = torchvision.transforms.ToTensor()
        self.transform_to_pil = torchvision.transforms.ToPILImage()

        if isfile("data/json2sg-%s" % identifier):
            print("Loading precomputed json2sg:", "data/json2sg-%s" % identifier)
            with open("data/json2sg-%s" % identifier, 'rb') as f:
                self.json2sg = pickle.load(f)
        else:
            self.json2sg = {}
            for js in self.scene_jsons:
                self.json2sg[js] = read_scene_json(js)  # given du['file_name']
            if if_save_data:
                with open("data/json2sg-%s" % identifier, 'wb') as f:
                    pickle.dump(self.json2sg, f, pickle.HIGHEST_PROTOCOL)

        if json2im is None:
            self.json2im = self.load_json2im(nb_samples=nb_samples)
        else:
            self.json2im = json2im

        keys = list(self.json2im.keys())
        if train:
            self.data = {du3: self.json2im[du3] for du3 in keys[:int(len(keys)*train_size)]}
        else:
            self.data = {du3: self.json2im[du3] for du3 in keys[int(len(keys)*train_size):]}
        self.keys = list(self.data.keys())
        print("loaded", len(self.json2im))

    def load_json2im(self, nb_samples=1000):
        if nb_samples < 0:
            nb_samples = len(self.scene_jsons)
        name = "%s-%d" % (self.root_dir.split("/")[-1], nb_samples)
        if isfile("data/%s" % name):
            print("Loading precomputed json2im:", "data/%s" % name)
            with open("data/%s" % name, 'rb') as f:
                return pickle.load(f)
        else:
            res_dict = {}
            if nb_samples > 0:
                random.shuffle(self.scene_jsons)
            for item in range(len(self.scene_jsons))[:nb_samples]:
                '''
                ALL ORIGINAL INFORMATION
                '''
                bboxes, coords, obj_names, img_name, d3coords = self.json2sg[self.scene_jsons[item]] # given a json dir --> read the json file 
                
                '''
                Reconstruct the scene graph
                '''
                
                sg = recon_sg2(self.scene_jsons[item])

                img_pil = Image.open("%s/%s" % (self.image_dir, img_name)).convert('RGB')
                img = self.transform(img_pil).unsqueeze(0)
                all_inp = []
                original_inp = [] # default mask
                weight_maps = [] # to get weight map of masks
                '''
                Get default mask and mask
                '''
                for bbox in bboxes:
                    mask_im = torch.zeros_like(img) # store original masks
                    default_inp = torch.zeros_like(img)

                    bbox_int = [int(dm11) for dm11 in bbox]

                    # masked the image
                    for x_ in range(bbox_int[0], bbox_int[2]+1):
                        idc = range(bbox_int[1], bbox_int[3]+1)
                        mask_im[:, :, idc, x_] = img[:, :, idc, x_]

                    # place the mask at the origin
                    idc1 = range(bbox_int[0], bbox_int[2]+1)
                    idc2 = range(len(idc1)) # for default
                    for j_, y_ in enumerate(range(bbox_int[1], bbox_int[3]+1)):
                        default_inp[:, :, j_+128-int(-bbox[1]+bbox[3]+1), idc2] = img[:, :, y_, idc1]

                    weight = construct_weight_map(bbox)

                    all_inp.append(mask_im)
                    original_inp.append(default_inp)
                    weight_maps.append(weight)

                targets = torch.from_numpy(np.array(coords)).float().flatten()
                all_inp = torch.cat(all_inp, dim=0).unsqueeze(0)
                original_inp = torch.cat(original_inp, dim=0).unsqueeze(0)
                weight_maps = torch.cat(weight_maps, dim=0).unsqueeze(0)
                '''
                all_inp: original mask image
                original_inp : default masks
                weight_maps: weight of original masks
                target: coordinates of masks
                obj_names: objects' names
                sg: scene grapph
                (all_inp, targets, original_inp, weight_maps, sg, obj_names)
                '''

                res_dict[self.scene_jsons[item]] = (all_inp, targets, original_inp, weight_maps, sg, obj_names, d3coords)
            with open("data/%s" % name, 'wb') as f:
                pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)
            return res_dict

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        return self.data[self.keys[item]]











class PBW_Planning_only(Dataset):
    '''
    (all_inp, sg, obj_names)
    loading data just for planning
    '''
    def __init__(self, root_dir=root_dir_PBWrandom_loc,
                 train=True, train_size=0.6, nb_samples=1000, json2im=None, base=True):
        print("Loading from", root_dir)
        super(PBW_Planning_only, self).__init__()
        self.root_dir = root_dir
        self.train = train
        self.base = base
        identifier = root_dir.split("/")[-1]+"planning_only"
        self.identifier = identifier
        json_dir = "%s/scene" % root_dir
        self.scene_jsons = [join(json_dir, f) for f in listdir(json_dir) if isfile(join(json_dir, f))]

        self.image_dir = "%s/image" % root_dir
        self.transform = torchvision.transforms.ToTensor()
        self.transform_to_pil = torchvision.transforms.ToPILImage()

        if isfile("data/json2sg-%s" % identifier):
            print("Loading precomputed json2sg:", "data/json2sg-%s" % identifier)
            with open("data/json2sg-%s" % identifier, 'rb') as f:
                self.json2sg = pickle.load(f)
        else:
            self.json2sg = {}
            for js in self.scene_jsons:
                self.json2sg[js] = read_scene_json(js)
            with open("data/json2sg-%s" % identifier, 'wb') as f:
                pickle.dump(self.json2sg, f, pickle.HIGHEST_PROTOCOL)

        if json2im is None:
            self.json2im = self.load_json2im(nb_samples=nb_samples)
        else:
            self.json2im = json2im

        keys = list(self.json2im.keys())
        if train:
            self.data = {du3: self.json2im[du3] for du3 in keys[:int(len(keys)*train_size)]}
        else:
            self.data = {du3: self.json2im[du3] for du3 in keys[int(len(keys)*train_size):]}
        self.keys = list(self.data.keys())
        print("loaded", len(self.json2im))

    def load_json2im(self, nb_samples=1000):
        if nb_samples < 0:
            nb_samples = len(self.scene_jsons)
        name = "%s-%d-planning" % (self.root_dir.split("/")[-1], nb_samples)
        if isfile("data/%s" % name):
            print("Loading precomputed json2im:", "data/%s" % name)
            with open("data/%s" % name, 'rb') as f:
                return pickle.load(f)
        else:
            res_dict = {}
            for item in range(len(self.scene_jsons))[:nb_samples]:
                bboxes, coords, obj_names, img_name, _ = self.json2sg[self.scene_jsons[item]]
                sg = recon_sg2(self.scene_jsons[item], if_add_bases=self.base)

                img_pil = Image.open("%s/%s" % (self.image_dir, img_name)).convert('RGB')
                img = self.transform(img_pil).unsqueeze(0)
                all_inp = []

                for bbox in bboxes:
                    mask_im = torch.zeros_like(img)
                    bbox_int = [int(dm11) for dm11 in bbox]

                    # masked the image
                    for x_ in range(bbox_int[0], bbox_int[2]+1):
                        idc = range(bbox_int[1], bbox_int[3]+1)
                        mask_im[:, :, idc, x_] = img[:, :, idc, x_]

                    all_inp.append(mask_im)

                all_inp = torch.cat(all_inp, dim=0).unsqueeze(0)  # masks

                res_dict[self.scene_jsons[item]] = (all_inp, sg, obj_names)
            with open("data/%s" % name, 'wb') as f:
                pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)
            return res_dict

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        return self.data[self.keys[item]]





class PBW_3D(Dataset):
    "loading Photorealistic data with true 3D information "
    def __init__(self, root_dir=root_dir_PWD3D,
                 train=True, train_size=0.6, nb_samples=-1, json2im=None, base=True):
        print("Loading from", root_dir)
        super(PBW_3D, self).__init__()
        self.root_dir = root_dir
        self.train = train
        self.base = base
        identifier = root_dir.split("/")[-1]+"3d_rel"
        self.identifier = identifier

        json_dir = "%s/scene" % root_dir
        self.scene_jsons = [join(json_dir, f) for f in listdir(json_dir)
                            if isfile(join(json_dir, f)) and "cam" not in f]

        self.image_dir = "%s/image" % root_dir
        self.transform = torchvision.transforms.ToTensor()
        self.transform_to_pil = torchvision.transforms.ToPILImage()

        if isfile("data/json2sg-%s" % identifier):
            print("Loading precomputed json2sg:", "data/json2sg-%s" % identifier)
            with open("data/json2sg-%s" % identifier, 'rb') as f:
                self.json2sg = pickle.load(f)
        else:
            self.json2sg = {}
            for js in self.scene_jsons:
                self.json2sg[js] = self.read_scene_json(js)
            with open("data/json2sg-%s" % identifier, 'wb') as f:
                pickle.dump(self.json2sg, f, pickle.HIGHEST_PROTOCOL)

        if json2im is None:
            self.json2im = self.load_json2im(nb_samples)
        else:
            self.json2im = json2im

        keys = list(self.json2im.keys())
        if train:
            self.data = {du3: self.json2im[du3] for du3 in keys[:int(len(keys)*train_size)]}
        else:
            self.data = {du3: self.json2im[du3] for du3 in keys[int(len(keys)*train_size):]}
        self.keys = list(self.data.keys())
        print("loaded", len(self.json2im))

    def load_json2im(self, nb_samples):
        if nb_samples < 0:
            nb_samples = len(self.scene_jsons)
        name = "%s-%d-%s" % (self.root_dir.split("/")[-1], nb_samples, self.identifier)
        if isfile("data/%s" % name):
            print("Loading precomputed json2im:", "data/%s" % name)
            with open("data/%s" % name, 'rb') as f:
                return pickle.load(f)
        else:
            res_dict = {}
            sg2behind = {}
            for item in range(len(self.scene_jsons))[:nb_samples]:
                '''
                self.json2sg[idx] contains bboxes, coords, obj_names, img_name
                '''
                bboxes, coords, obj_names, img_name = self.json2sg[self.scene_jsons[item]]

                sg, sh_hash, front_objects, behind_objects, base_dict = self.recon_sg(coords, obj_names)

                if len(front_objects) == 0:
                    sg2behind[sh_hash] = (None, None, img_name)
                    continue

                sg_n1 = []
                sg_n2 = []
                for i_ in range(len(front_objects)-1):
                    sg_n1.append([front_objects[i_], "up", front_objects[i_+1]])
                    sg_n2.append([front_objects[i_], "up", front_objects[i_+1]])
                for rel in sg:
                    if rel[0] not in front_objects and rel[2] not in front_objects:
                        sg_n1.append(rel)
                        sg_n2.append(rel)

                sg_n1.append([front_objects[-1], "up", "00"])
                if "10" in base_dict:
                    sg_n1.append([front_objects[-1], "front", base_dict["10"]])
                sg_n2.append([front_objects[-1], "up", "02"])
                if "12" in base_dict:
                    sg_n2.append([front_objects[-1], "front", base_dict["12"]])


                img_pil = Image.open("%s/%s" % (self.image_dir, img_name)).convert('RGB')
                img = self.transform(img_pil).unsqueeze(0)
                front_masks = []
                behind_masks = []
                front_masks_name = []
                behind_masks_name = []
                for i_, bbox in enumerate(bboxes):
                    mask_im = torch.zeros_like(img)
                    bbox_int = [int(dm11) for dm11 in bbox]

                    # masked the image
                    for x_ in range(bbox_int[0], bbox_int[2]+1):
                        idc = range(bbox_int[1], bbox_int[3]+1)
                        mask_im[:, :, idc, x_] = img[:, :, idc, x_]

                    if coords[i_][1] == -2:
                        front_masks.append(mask_im)
                        front_masks_name.append(obj_names[i_])
                    elif coords[i_][1] == 2 and obj_names[i_] in behind_objects: # separate bounding box
                        behind_masks.append(mask_im)
                        behind_masks_name.append(obj_names[i_])


                if front_masks:
                    front_masks = torch.cat(front_masks, dim=0).unsqueeze(0)  # masks
                else:
                    front_masks = []

                if behind_masks:
                    behind_masks = torch.cat(behind_masks, dim=0).unsqueeze(0)  # masks
                else:
                    behind_masks = []

                assert sh_hash not in sg2behind, "sg hash code not unique"
                sg2behind[sh_hash] = (behind_masks, behind_objects, img_name)
                ##sg_n1: up relation 
                ##sg_n2: forward relation 
               
                res_dict[self.scene_jsons[item]] = (sg, obj_names, front_masks, behind_masks, behind_objects, sg_n1, sg_n2,
                                                    img_name, front_masks_name, behind_masks_name)
            res_dict["sg2behind"] = sg2behind

            with open("data/%s" % name, 'wb') as f:
                pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)
            return res_dict


    def recon_sg(self, loc_, name_): #00 is y-stack ID 0 and x stack ID 0 
        front_ = []
        behind_ = []
        name2loc = {n: loc_[i_] for i_, n in enumerate(name_)}
        bases = {"-3": "0", "0": "1",  "3": "2"}  # x-coords: stack ID
        base_sg = []
        front_objects = []
        behind_objects = []
        '''
        The only possible y-coords are -2 and 2
        '''
        base_dict_all = {}
        for i_, n in enumerate(name_):
            if loc_[i_][1] == -2: # the y-coordinates
                base_sg.append([n, "up", "0"+bases[str(int(loc_[i_][0]))]]) #the objetcts' x-cord
                front_.append(n)
                front_objects.append(n)
                base_dict_all[n] = "0" + bases[str(int(loc_[i_][0]))]
            elif loc_[i_][1] == 2:
                base_sg.append([n, "up", "1"+bases[str(int(loc_[i_][0]))]])
                behind_.append(n)
                behind_objects.append(n)
                base_dict_all[n] = "1" + bases[str(int(loc_[i_][0]))]

        for obj in behind_objects[:]:
            if base_dict_all[obj][0] == "1": # y stack ID
                front_base = "0"+base_dict_all[obj][1] # move it to forward 
                if front_base in base_dict_all.values():
                    behind_objects.remove(obj)

        behind_sg = []
        front_sg = []
        '''
        find the front and behind up relation separately 
        '''
        if len(front_) > 0:
            front_sg = recon_sg(front_, [(name2loc[n][0], name2loc[n][2]) for n in front_], if_add_bases=False)
        if len(behind_) > 0:
            behind_sg = recon_sg(behind_, [(name2loc[n][0], name2loc[n][2]) for n in behind_], if_add_bases=False) # Now there is no overlap we can use recon_sg
                        # returen up relation 
        base_dict = {}
        for rel in base_sg[:]:
            if rel[0] in [du[0] for du in (front_sg+behind_sg)]: # delete redundant relations
                base_sg.remove(rel)
        for rel in base_sg:
            assert rel[2] not in base_dict # rel[2] is the base
            base_dict[rel[2]] = rel[0] # base_dict is base:color; base_dict_all is color: base

        total_sg = front_sg+behind_sg+base_sg

        line = {du1: [] for du1 in ["00", "01", "02", "10", "11", "12"]}
        for rel in total_sg:
            if rel[2] in ["00", "01", "02", "10", "11", "12"]:
                line[rel[2]].append(rel[0]) # The base distribution 
        '''
        The Front relation 
        '''
        threed_sg = []
        for base in base_dict:
           if base[0] == "0":
               if str(int(base)+10) in base_dict:
                   threed_sg.append([base_dict[base], "front", base_dict[str(int(base)+10)]])

        total_sg = front_sg+behind_sg+base_sg+threed_sg
        return total_sg, self.hash_sg(total_sg), front_objects, behind_objects, base_dict


    def read_scene_json(self, json_file_dir):
        '''
        EDIT KE original --  'blue' : [42,75,215]  "cyan": [41, 208, 208]
        CHANGE THE CORS
        '''
        id2color = {
            "gray": [87, 87, 87],
            "red": [173, 35, 35],
            "blue": [42,75,215],
            "green": [29, 105, 20],
            "brown": [129, 74, 25],
            "purple": [129, 38, 192],
            "cyan": [41, 208, 208],
            "yellow": [255, 238, 51],
            "c1": [42, 87, 9],
            "c2": [255, 102, 255],
            "orange": [255, 140, 0],
            "turquoise": [50,168, 149]
        }
        
        
        color2id = {tuple(v): u for u, v in id2color.items()}
        # print(json_file_dir)
        with open(json_file_dir, 'r') as json_file:
            du = json.load(json_file)
        location_dict = {}
        objects = []
        bboxes = []
        locations = []
        shapes = []

        for obj in du["objects"]:
            color = tuple([int(du33 * 255) for du33 in obj["color"]][:-1])
            object_id = color2id[color]
            a_key = "%.3f" % obj["location"][0]
            if a_key not in location_dict:
                location_dict[a_key] = [(object_id, obj["location"][2])]
            else:
                location_dict[a_key].append((object_id, obj["location"][2]))
            objects.append(object_id)
            bboxes.append([
                obj["bbox"][0],
                obj["bbox"][1],
                obj["bbox"][2],
                obj["bbox"][3],
            ])
            locations.append([obj["location"][0], obj["location"][1], obj["location"][2]]) # CONTAINS ALL LOCATIONS 
            shapes.append(obj["shape"])

        return bboxes, locations, objects, du["image_filename"]




## CHANGE HERE


    def hash_sg(self, relationships,
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

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        return self.data[self.keys[item]]









class PBW_3D_est(Dataset):
    "loading Photorealistic data with our estimated 3D information "
    def __init__(self, root_dir=root_dir_PWD3D,
                 train=True, train_size=0.6, nb_samples=-1, json2im=None, base=True):
        print("Loading from", root_dir)
        super(PBW_3D_est, self).__init__()
        # KE
        self.m = 0
        
        self.root_dir = root_dir
        self.train = train
        self.base = base
        identifier = root_dir.split("/")[-1]+"3d_rel_est"
        self.identifier = identifier

        json_dir = "%s/scene" % root_dir
        self.scene_jsons = [join(json_dir, f) for f in listdir(json_dir)
                            if isfile(join(json_dir, f)) and "cam" not in f]

        self.image_dir = "%s/image" % root_dir
        self.transform = torchvision.transforms.ToTensor()
        self.transform_to_pil = torchvision.transforms.ToPILImage()

        if isfile("data/json2sg-%s" % identifier):
            print("Loading precomputed json2sg:", "data/json2sg-%s" % identifier)
            with open("data/json2sg-%s" % identifier, 'rb') as f:
                self.json2sg = pickle.load(f)
        else:
            self.json2sg = {}
            for js in self.scene_jsons:
                self.json2sg[js] = self.read_scene_json(js)
            with open("data/json2sg-%s" % identifier, 'wb') as f:
                pickle.dump(self.json2sg, f, pickle.HIGHEST_PROTOCOL)

        if json2im is None:
            self.json2im = self.load_json2im(nb_samples)
        else:
            self.json2im = json2im

        keys = list(self.json2im.keys())
        if train:
            self.data = {du3: self.json2im[du3] for du3 in keys[:int(len(keys)*train_size)]}
        else:
            self.data = {du3: self.json2im[du3] for du3 in keys[int(len(keys)*train_size):]}
        self.keys = list(self.data.keys())
        print("loaded", len(self.json2im))

    def load_json2im(self, nb_samples):
        # KE
        if nb_samples < 0:
            nb_samples = len(self.scene_jsons)
        name = "%s-%d-%s" % (self.root_dir.split("/")[-1], nb_samples, self.identifier)
        if isfile("data/%s" % name):
            print("Loading precomputed json2im:", "data/%s" % name)
            with open("data/%s" % name, 'rb') as f:
                return pickle.load(f)
        else:
            res_dict = {}
            sg2behind = {}
            for item in range(len(self.scene_jsons))[:nb_samples]:
                '''
                self.json2sg[idx] contains bboxes, coords, obj_names, img_name
                '''
                bboxes, coords, obj_names, img_name = self.json2sg[self.scene_jsons[item]]

                sg, sh_hash, front_objects, behind_objects, base_dict = self.recon_sg(coords, obj_names)

                if len(front_objects) == 0:
                    sg2behind[sh_hash] = (None, None, img_name)
                    continue

                sg_n1 = []
                sg_n2 = []
                for i_ in range(len(front_objects)-1):
                    sg_n1.append([front_objects[i_], "up", front_objects[i_+1]])
                    sg_n2.append([front_objects[i_], "up", front_objects[i_+1]])
                for rel in sg:
                    if rel[0] not in front_objects and rel[2] not in front_objects:
                        sg_n1.append(rel)
                        sg_n2.append(rel)

                sg_n1.append([front_objects[-1], "up", "00"])
                if "10" in base_dict:
                    sg_n1.append([front_objects[-1], "front", base_dict["10"]])
                sg_n2.append([front_objects[-1], "up", "02"])
                if "12" in base_dict:
                    sg_n2.append([front_objects[-1], "front", base_dict["12"]])


                img_pil = Image.open("%s/%s" % (self.image_dir, img_name)).convert('RGB')
                img = self.transform(img_pil).unsqueeze(0)
                front_masks = []
                behind_masks = []
                front_masks_name = []
                behind_masks_name = []
                for i_, bbox in enumerate(bboxes):
                    mask_im = torch.zeros_like(img)
                    bbox_int = [int(dm11) for dm11 in bbox]

                    # masked the image
                    for x_ in range(bbox_int[0], bbox_int[2]+1):
                        idc = range(bbox_int[1], bbox_int[3]+1)
                        mask_im[:, :, idc, x_] = img[:, :, idc, x_]

                    if coords[i_][1] == -2:
                        front_masks.append(mask_im)
                        front_masks_name.append(obj_names[i_])
                    elif coords[i_][1] == 2 and obj_names[i_] in behind_objects: # separate bounding box
                        behind_masks.append(mask_im)
                        behind_masks_name.append(obj_names[i_])


                if front_masks:
                    front_masks = torch.cat(front_masks, dim=0).unsqueeze(0)  # masks
                else:
                    front_masks = []

                if behind_masks:
                    behind_masks = torch.cat(behind_masks, dim=0).unsqueeze(0)  # masks
                else:
                    behind_masks = []
                ## EDIT By KE
                if sh_hash not in sg2behind:
                    sg2behind[sh_hash] = (behind_masks, behind_objects, img_name)
                else:
                    self.m = self.m+1
                    # print(sh_hash)
                
                #assert sh_hash not in sg2behind, "sg hash code not unique"
                #sg2behind[sh_hash] = (behind_masks, behind_objects, img_name)
                ##sg_n1: up relation 
                ##sg_n2: forward relation 
               
                res_dict[self.scene_jsons[item]] = (sg, obj_names, front_masks, behind_masks, behind_objects, sg_n1, sg_n2,
                                                    img_name, front_masks_name, behind_masks_name)
            res_dict["sg2behind"] = sg2behind

            with open("data/%s" % name, 'wb') as f:
                pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)

            return res_dict



    def recon_sg(self, loc_, name_): #00 is y-stack ID 0 and x stack ID 0 
        front_ = []
        behind_ = []
        name2loc = {n: loc_[i_] for i_, n in enumerate(name_)}
        bases = {"-3": "0", "0": "1",  "3": "2"}  # x-coords: stack ID
        base_sg = []
        front_objects = []
        behind_objects = []
        '''
        The only possible y-coords are -2 and 2
        '''
        base_dict_all = {}
        for i_, n in enumerate(name_):
            try:
                bases[str(int(loc_[i_][0]))]
            except:
                print(loc_[i_])
                
            if loc_[i_][1] == -2: # the y-coordinates
                base_sg.append([n, "up", "0"+bases[str(int(loc_[i_][0]))]]) #the objetcts' x-cord
                front_.append(n)
                front_objects.append(n)
                base_dict_all[n] = "0" + bases[str(int(loc_[i_][0]))]
            elif loc_[i_][1] == 2:
                base_sg.append([n, "up", "1"+bases[str(int(loc_[i_][0]))]])
                behind_.append(n)
                behind_objects.append(n)
                base_dict_all[n] = "1" + bases[str(int(loc_[i_][0]))]

        for obj in behind_objects[:]:
            if base_dict_all[obj][0] == "1": # y stack ID
                front_base = "0"+base_dict_all[obj][1] # move it to forward 
                if front_base in base_dict_all.values():
                    behind_objects.remove(obj)

        behind_sg = []
        front_sg = []
        '''
        find the front and behind up relation separately 
        '''
        if len(front_) > 0:
            front_sg = recon_sg(front_, [(name2loc[n][0], name2loc[n][2]) for n in front_], if_add_bases=False)
        if len(behind_) > 0:
            behind_sg = recon_sg(behind_, [(name2loc[n][0], name2loc[n][2]) for n in behind_], if_add_bases=False) # Now there is no overlap we can use recon_sg
                        # returen up relation 
        base_dict = {}
        for rel in base_sg[:]:
            if rel[0] in [du[0] for du in (front_sg+behind_sg)]: # delete redundant relations
                base_sg.remove(rel)
        for rel in base_sg:
            assert rel[2] not in base_dict # rel[2] is the base
            base_dict[rel[2]] = rel[0] # base_dict is base:color; base_dict_all is color: base

        total_sg = front_sg+behind_sg+base_sg

        line = {du1: [] for du1 in ["00", "01", "02", "10", "11", "12"]}
        for rel in total_sg:
            if rel[2] in ["00", "01", "02", "10", "11", "12"]:
                line[rel[2]].append(rel[0]) # The base distribution 
        '''
        The Front relation 
        '''
        threed_sg = []
        for base in base_dict:
           if base[0] == "0":
               if str(int(base)+10) in base_dict:
                   threed_sg.append([base_dict[base], "front", base_dict[str(int(base)+10)]])

        total_sg = front_sg+behind_sg+base_sg+threed_sg
        return total_sg, self.hash_sg(total_sg), front_objects, behind_objects, base_dict


    def read_scene_json(self, json_file_dir):
        '''
        EDIT KE original --  'blue' : [42,75,215]  "cyan": [41, 208, 208]
        CHANGE THE CORS
        '''
        id2color = {
            "gray": [87, 87, 87],
            "red": [173, 35, 35],
            "blue": [42,75,215],
            "green": [29, 105, 20],
            "brown": [129, 74, 25],
            "purple": [129, 38, 192],
            "cyan": [41, 208, 208],
            "yellow": [255, 238, 51],
            "c1": [42, 87, 9],
            "c2": [255, 102, 255],
            "orange": [255, 140, 0],
            "turquoise": [50,168, 149]
        }
        
        
        color2id = {tuple(v): u for u, v in id2color.items()}
        # print(json_file_dir)
        with open(json_file_dir, 'r') as json_file:
            du = json.load(json_file)


            
        with open('predicted_3D_coords', 'rb') as f:
                coordinates_3d = pickle.load(f)    
                
                
                
        location_dict = {}
        objects = []
        bboxes = []
        locations = []
        shapes = []

        for idx, obj in enumerate(du["objects"]):
            color = tuple([int(du33 * 255) for du33 in obj["color"]][:-1])
            object_id = color2id[color]
            a_key = "%.3f" % obj["location"][0]
            if a_key not in location_dict:
                location_dict[a_key] = [(object_id, obj["location"][2])]
            else:
                location_dict[a_key].append((object_id, obj["location"][2]))
            objects.append(object_id)
            bboxes.append([
                obj["bbox"][0],
                obj["bbox"][1],
                obj["bbox"][2],
                obj["bbox"][3],
            ])
            obj_location = coordinates_3d[json_file_dir][idx]
            locations.append([obj_location[0].item() , obj_location[1].item() , obj_location[2].item() ])
            # locations.append([obj["location"][0], obj["location"][1], obj["location"][2]]) # CONTAINS ALL LOCATIONS 
            shapes.append(obj["shape"])

        return bboxes, locations, objects, du["image_filename"]




## CHANGE HERE


    def hash_sg(self, relationships,
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

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        return self.data[self.keys[item]]





def construct_weight_map(bbox):
    '''
    Same as the one in utils
    '''
    weight_map = torch.ones(128, 128)
    a_ = 255
    b_ = 1
    bbox_int = [int(dm11) for dm11 in bbox]

    weight_map[:, range(0, bbox_int[0]+1)] *= torch.tensor([(b_-a_)*x_/bbox[0]+a_ for x_ in range(0, bbox_int[0]+1)])
    weight_map[:, range(bbox_int[2], 128)] *= torch.tensor([(a_-b_)*(x_-bbox[2])/(128-bbox[2])+b_ for x_ in range(bbox_int[2], 128)])

    for x_ in range(0, bbox_int[1]+1):
        weight_map[x_, :] *= (b_-a_)*x_/bbox[1]+a_
    for x_ in range(bbox_int[3], 128):
        weight_map[x_, :] *= (a_-b_)*(x_-bbox[3])/(128-bbox[3])+b_

    for x_ in range(bbox_int[0], bbox_int[2] + 1):
        for y_ in range(bbox_int[1], bbox_int[3] + 1):
            weight_map[y_, x_] = 0

    weight_map = torch.sqrt(weight_map)
    return weight_map.unsqueeze(0)







def recon_sg2(json_file_dir, if_add_bases=True):
    
    """
    WITHOUT OVERLAPPING
    
    
    reconstruct a sg from a scene json file
    EDIT KE origianl  ; "cyan": [41, 208, 208] 
    """
    id2color = {
        "gray": [87, 87, 87],
        "red": [173, 35, 35],
        "blue": [42,75,215],
        "green": [29, 105, 20],
        "brown": [129, 74, 25],
        "purple": [129, 38, 192],
        "cyan": [41, 208, 208],
        "yellow": [255, 238, 51],
        "c1": [42, 87, 9],
        "c2": [255, 102, 255],
        "orange": [255, 140, 0],
        "turquoise": [50,168, 149]
    }

    color2id = {tuple(v): u for u, v in id2color.items()}
    with open(json_file_dir, 'r') as json_file:
        du = json.load(json_file)
    location_dict = {} # tracing x-cord
    objects = []
    bboxes = []
    for obj in du["objects"]:
        color = tuple([int(du33*255) for du33 in obj["color"]][:-1])
        object_id = color2id[color]
        a_key = "%.3f" % obj["location"][0] # x-cord
        if a_key not in location_dict:
            location_dict[a_key] = [(object_id, obj["location"][2])]  # z- cord 
        else:
            location_dict[a_key].append((object_id, obj["location"][2]))
        objects.append(object_id)
        bboxes.append([
            obj["bbox"][0]/128.0,  # scale down
            obj["bbox"][1]/128.0,
            obj["bbox"][2]/128.0,
            obj["bbox"][3]/128.0,
            ])
        
    obj2id = {objects[du4]: objects[du4] for du4 in range(len(objects))}
    
    if if_add_bases:
        
        '''
        EDIT KE, Original 
        
        relationships = [
            [obj2id["brown"], "left", obj2id["purple"]],
            [obj2id["purple"], "left", obj2id["cyan"]],
        ]
        '''
        relationships = [
            ["brown", "left", "purple"],
            ["purple", "left", "cyan"],
        ]
    else:
        relationships = []
    for du3 in location_dict:
        location = sorted(location_dict[du3], key=lambda x: x[1])
        while len(location) > 1:
            o1 = location.pop()[0]
            o2 = location[-1][0]
            relationships.append([obj2id[o1], "up", obj2id[o2]])
            assert o1 not in ["cyan", "purple", "brown"]

    return relationships



def recon_sg(obj_names, locations, if_return_assigns=False, if_add_bases=True):
    """
    reconstruct a sg from object names and coordinates
    """
    location_dict = {}
    objects = []

    # change the type of locations
    if type(locations) == torch.Tensor:
        locations = locations.cpu().numpy()
    elif isinstance(locations, list):
        locations = np.array(locations)

    locations = locations.reshape(-1, 2)
    k_means_assign = kmeans(locations[:, 0])
    
    '''
     Divide objects into n stacks
    '''
    for idx, object_id in enumerate(obj_names):
        a_key = k_means_assign[idx] # a_key the stack ID
        if a_key not in location_dict:
            location_dict[a_key] = [(object_id, locations[idx][1])]  # the z-coordinates
        else:
            location_dict[a_key].append((object_id, locations[idx][1]))
        objects.append(object_id)  # object_id = obj_name
    relationships = []
    
    
    if if_add_bases:
        relationships.extend([
            ["brown", "left", "purple"],
            ["purple", "left", "cyan"],
        ])
    '''
    the construction of up relation might miss something
    '''    
    for du3 in location_dict:
        location = sorted(location_dict[du3], key=lambda x: x[1])
        while len(location) > 1:
            o1 = location.pop()[0]
            o2 = location[-1][0]
            relationships.append([o1, "up", o2])
    if if_return_assigns:
        return relationships, k_means_assign
    return relationships






def check(d1, d2):
    '''
    checking wehether two objects are same or not 
    '''
    for i in range(len(d1)):
        if d1[i] != d2[i]:
            print(d1)
            print(d2)
            print(d1[i], d2[i], i)
            return False
    return True




def sim_collate_fn(batch):
    '''
    batch --> [masks, def_mat, def_wei, ob_names, sg, im_name]
    
    '''
    all_imgs, all_imgs2, weights = [], [], []
    sgs = []
    names = []
    im_names = []
    for masks, def_mat, def_wei, ob_names, sg, im_name in batch:
        all_imgs.append(masks.unsqueeze(0))
        all_imgs2.append(def_mat.unsqueeze(0))
        weights.append(def_wei.unsqueeze(0))
        sgs.append(sg)
        names.append(ob_names)
        im_names.append(im_name)

    all_imgs = torch.cat(all_imgs)
    all_imgs2 = torch.cat(all_imgs2)
    weights = torch.cat(weights)

    return all_imgs, all_imgs2, weights, sgs, names, im_names




def pbw_collate_fn(batch):
    '''
    batch --> [original in]
    all_imgs: inp(original masks), tar(coordinates), inp2(default masks), sg(scene graph), obj_name (objects' names)
        
                all_inp: original mask image
                original_inp : default masks
                weight_maps: weight of original masks
                target: coordinates of masks
                obj_names: objects' names
                sg: scene grapph
                (all_inp, targets, original_inp, weight_maps, sg, obj_names)
                    
    '''
    all_imgs, all_targets, all_imgs2, weights = [], [], [], []
    sgs = []
    names = []
    for i, (inp, tar, inp2, w, sg, name) in enumerate(batch):
        all_imgs.append(inp)
        all_imgs2.append(inp2)
        all_targets.append(tar.unsqueeze(0))
        weights.append(w)
        sgs.append(sg)
        names.append(name)

    all_imgs = torch.cat(all_imgs)
    all_imgs2 = torch.cat(all_imgs2)
    all_targets = torch.cat(all_targets)
    weights = torch.cat(weights)

    return all_imgs, all_targets, all_imgs2, weights, sgs, names




def collate_fn_trans(batch):
    '''
    batch --> start configuration , action, end configurtion 
    '''
    all_imgs1, all_imgs2, all_act = [], [], []
    for start, act_onehot, end in batch:
        all_imgs1.append(start.unsqueeze(0))
        all_imgs2.append(end.unsqueeze(0))
        all_act.append(act_onehot.unsqueeze(0))

    all_imgs1 = torch.cat(all_imgs1)
    all_imgs2 = torch.cat(all_imgs2)
    all_act = torch.cat(all_act).float()

    return all_imgs1, all_imgs2, all_act






def evaluation(json2im, model, loss_func, device="cuda"):
    '''
    
    difference between orginal sg and reconstructed sd
    '''
    val_loss = []
    correct = 0
    total = 0.0
    best_p = None
    best_l = None
    best_a = None
    print("evaluating %d samples" % len(json2im))

    all_images = []
    for val_json in json2im:
        val_batch = json2im[val_json][:2]
        images, coords = [tensor.to(device).double() for tensor in val_batch]
        all_images.append(images)
    all_images = torch.cat(all_images, dim=0)
    im_iter = DataLoader(all_images, batch_size=64)
    all_pred_coords = []
    for im in im_iter:
        with torch.no_grad():
            pred_coords = model(im)
            all_pred_coords.append(pred_coords)
    all_pred_coords = torch.cat(all_pred_coords, dim=0)

    for idx, val_json in enumerate(json2im):
        val_batch = json2im[val_json][:2]
        obj_names = json2im[val_json][2]
        images, coords = [tensor.to(device).double() for tensor in val_batch]
        sg = recon_sg(obj_names, coords)

        pred_coords = all_pred_coords[idx]
        pred_sg, assigns = recon_sg(obj_names, pred_coords, if_return_assigns=True) # assign --> kmeans()
        correct += sg == pred_sg # if sg = pred_sg, correct add one
        total += 1
        loss = loss_func(pred_coords, coords)
        val_loss.append(loss.item())
        if best_l is None or loss.item() < best_l:
            best_l = loss.item()
            best_p = (pred_coords, coords)
            best_a = assigns
    print("acquire this best:")
    print("assigns", best_a)
    print("pred\n", best_p[0])
    print("true\n", best_p[1])
    print()
    acc = correct / total
    return np.mean(val_loss), acc





def evaluation2(iter_, model, loss_func, device="cuda"):
    '''
    average loss of a batch of images
    iter_ --> batch of [images, coords]
    '''
    val_loss = []
    for val_batch in iter_:
        images, coords = [tensor.to(device).double() for tensor in val_batch]

        with torch.no_grad():
            pred_coords = model(images)

        loss = loss_func(pred_coords, coords)
        val_loss.append(loss.item())
    return np.mean(val_loss)





def find_top(up_rel, ob_start):
    '''
    find upper
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





def read_scene_json(json_file_dir, return_top_objects=False, return_object_shape=False ):
    from random import shuffle
    '''
    EDIT BY KE orginal :    'blue ': [42,75, 215], "cyan": [41, 208, 208],
    '''
    id2color = {
        "gray": [87, 87, 87],
        "red": [173, 35, 35],
        "blue": [42,75, 215],
        "green": [29, 105, 20],
        "brown": [129, 74, 25],
        "purple": [129, 38, 192],
        "cyan": [41, 208, 208],
        "yellow": [255, 238, 51],
        "c1": [42, 87, 9],
        "c2": [255, 102, 255],
        "orange": [255, 140, 0],
        "turquoise": [50,168, 149]
    }
    color2id = {tuple(v): u for u, v in id2color.items()}
    with open(json_file_dir, 'r') as json_file:
        du = json.load(json_file)
    location_dict = {}
    objects = []
    bboxes = []
    locations = []
    d3location = []
    shapes = []

    for obj in du["objects"]:
        color = tuple([int(du33*255) for du33 in obj["color"]][:-1])
        object_id = color2id[color]
        a_key = "%.3f" % obj["location"][0] # the x-cordinate 
        if a_key not in location_dict:
            location_dict[a_key] = [(object_id, obj["location"][2])] # the highet of objects
        else:
            location_dict[a_key].append((object_id, obj["location"][2]))
            
        objects.append(object_id) # object_id color of the object 
        #bbox x1, y1, x2, y2
        bboxes.append([
            obj["bbox"][0],
            obj["bbox"][1],
            obj["bbox"][2],
            obj["bbox"][3],
        ])
        d3location.append([obj["location"][0], obj["location"][1],obj["location"][2]])
        locations.append([obj["location"][0], obj["location"][2]]) # the x-coord and the hight z-coord
        shapes.append(obj["shape"]) # objects' shape


    if return_top_objects:
        '''
        DOES NOT WORK FOR 3D
        '''
        sg = recon_sg(objects, locations) # 'objects' contains the objects' names
        up_rel = [rel for rel in sg if rel[1] == "up"]
        values_for_masks = [dm1 + 1 for dm1 in list(range(len(objects)))]
        shuffle(values_for_masks)
        obID2mask_value = {objects[dm3]: values_for_masks[dm3] for dm3 in range(len(objects))}
        ob_mask_value = [obID2mask_value[dm4] for dm4 in objects]
        
        '''
        CAREFUL THIS ONE REQUIRES THE BASE 'brown', 'purple', 'cayan'
        '''
        
        top_obj_ids = [obID2mask_value[find_top(up_rel, obj)[0]] - 1 for obj in ["brown", "purple", "cyan"]]
        return bboxes, locations, objects, du["image_filename"], ob_mask_value, top_obj_ids
    if return_object_shape:
        return bboxes, locations, objects, du["image_filename"], shapes

    return bboxes, locations, objects, du["image_filename"], d3location





def kmeans(data_):
    '''
    # Don't require the # of stacks
    Assuning there is three stack 
    '''
    c1 = max(data_)
    c2 = min(data_)
    c3 = (c1+c2)/2
    c_list = [c1, c2, c3]
    init_c_list = c_list[:]
    assign = [0]*len(data_)
    # print(data_)
    # print(data_.shape)
    for _ in range(10):
        for idx, d in enumerate(data_):
            assign[idx] = min([0, 1, 2], key=lambda x: (c_list[x]-d)**2)

        for c in range(3):
            stuff = [d for idx, d in enumerate(data_) if assign[idx] == c]
            if len(stuff) > 0:
                c_list[c] = sum(stuff)/len(stuff)
    # print(sorted(c_list), sorted(init_c_list), data_)
    return assign





if __name__ == '__main__':
    # d = PBW_3D()
    PBW_all = PBW(nb_samples=-1, train_size=1)
    #est_data = PBW_3D_est(train_size=1.0, nb_samples=-1)
    
    
# for ele in coordinates_3d:
#     for i in range(5):
#         if ele[i][0] not in {-3,0,3}:
#             print(ele)
#         if ele[i][1] not in {-2, 2}:    
#             print(ele)
            
# for ele in coordinates_3d.values():
#     for i in range(5):
#         if ele[i][0] not in {-3,0,3}:
#             print(ele)
#         if ele[i][1] not in {-2, 2}:    
#             print(ele)    
