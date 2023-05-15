import torch
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
import numpy as np
import torchvision
import time
import more_itertools
from torchvision.utils import make_grid
from PIL import Image


dir_read_seg_masks_slow = '/Users/pippala/Desktop/robotic/location-based-generative-master/5objs_seg/z.seg834_s1_4123_s2__s0_0.ppm'


def construct_weight_map(bbox):
    '''
    bbox: boundaries of boundingbox
    '''
    weight_map = torch.ones(128, 128)
    a_ = 255
    b_ = 1
    bbox_int = [int(dm11) for dm11 in bbox]

    weight_map[:, range(0, bbox_int[0]+1)] *= torch.tensor([(b_-a_)*x_/bbox[0]+a_ for x_ in range(0, bbox_int[0]+1)]) # leftmost column, decreasing to 1
    weight_map[:, range(bbox_int[2], 128)] *= torch.tensor([(a_-b_)*(x_-bbox[2])/(128-bbox[2])+b_ for x_ in range(bbox_int[2], 128)]) #rightmost column, increasing from 1


    for x_ in range(0, bbox_int[1]+1): #upest, decreasing to 1
        weight_map[x_ , :] *= (b_-a_)*x_/bbox[1]+a_  # origianl weight_map[x_, :] *= (b_-a_)*x_/bbox[1]+a_ 
        
    for x_ in range(bbox_int[3], 128): #lowest, increasing from 1
        weight_map[x_ , :] *= (a_-b_)*(x_-bbox[3])/(128-bbox[3])+b_

    for x_ in range(bbox_int[0], bbox_int[2] + 1): # ele in the bounding box
        for y_ in range(bbox_int[1], bbox_int[3] + 1):
            weight_map[y_, x_] = 0

    weight_map = torch.sqrt(weight_map)
    return weight_map.unsqueeze(0) # 1* 128*128, one mask



def compute_grad(model_):
    res = 0
    for param in model_.parameters():
        if param.requires_grad:
            if param.grad is not None:
                res += abs(torch.sum(torch.abs(param.grad)).item())
    return res


def show2(im_, name, nrow):
    '''
    for example show2([masks, def_mat, wei_mat], "masks_test", 5)
    show the masks, def_mat, and wei_mat images
    '''
    import logging

    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)

    fig_ = plt.figure(figsize=(15, 15))
    for du3 in range(1, len(im_)+1):
        plt.subplot(1, len(im_), du3)
        plt.axis("off")
        plt.imshow(np.transpose(make_grid(im_[du3-1], padding=5, normalize=False, pad_value=50, nrow=nrow),
                                (1, 2, 0)))

    plt.axis("off")
    # plt.title("black: no action, red: 1-3, yellow: 3-1, green: 1-2, blue: 2-3, pink: 3-2, brown: 2-1")
    plt.savefig(name, transparent=True, bbox_inches='tight')
    print("saved to", name)
    plt.close(fig_)
    logger.setLevel(old_level)


def recon_sg(obj_names, locations, nb_clusters, if_return_assigns=False):
    """
    reconstruct a scene graph from object names and coordinates
    
    locations : coordinates of each object
    
    nb_cluster applied in kmean to cluster objects
    
    """
    location_dict = {}
    objects = []

    if type(locations) == torch.Tensor:
        locations = locations.cpu().numpy()
    elif isinstance(locations, list):
        locations = np.array(locations)

    locations = locations.reshape(-1, 2) # convert to 2D
    k_means_assign = kmeans(locations[:, 0], nb_clusters) #cluster by x-axis

    if nb_clusters == 1:
        k_means_assign = [0]*len(obj_names)

    for idx, object_id in enumerate(obj_names):
        a_key = k_means_assign[idx]
        if a_key not in location_dict:
            location_dict[a_key] = [(object_id, locations[idx][1], locations[idx][0])]
        else:
            location_dict[a_key].append((object_id, locations[idx][1], locations[idx][0]))
        objects.append(object_id)
    relationships = []

    # decide up relation
    bottoms = []
    for du3 in location_dict:
        location = sorted(location_dict[du3], key=lambda x: x[1])
        bottoms.append(location[0])
        while len(location) > 1:
            o1 = location.pop()[0]
            o2 = location[-1][0]
            relationships.append([o1, "up", o2])

    # decide left relation
    bottoms = sorted(bottoms, key=lambda x: x[2])

    if len(bottoms) > 1:
        relationships.append([bottoms[0][0], "left", bottoms[1][0]])
    if len(bottoms) > 2:
        relationships.append([bottoms[1][0], "left", bottoms[2][0]])
    if if_return_assigns:
        return relationships,

    return relationships




def find_nb_blocks(scene_image):
    '''
    scene_image a layer of a image, generally first layer
    '''
    all_res = []
    for i in range(127, -1, -1):
        indices = scene_image[i].nonzero().squeeze().cpu().numpy() # for each row checking pixel's value
        try:
            if len(indices) > 0:
                res = [list(group) for group in more_itertools.consecutive_groups(indices)] # consecutive number
                if len(res) == 3: # at most 3 stacks 
                    return 3
                else:
                    all_res.append(len(res))
        except TypeError:
            pass
    return max(all_res)



def kmeans(data_, nb_clusters):
    """
    assign each object one of 3 block IDs based on the x coord
    :param data_:
    :return:
    """
    c1 = max(data_)
    c2 = min(data_)
    c3 = (c1+c2)/2
    c_list = [c2, c3, c1]
    init_c_list = c_list[:]
    assign = [0]*len(data_)

    for _ in range(10):
        for idx, d in enumerate(data_):
            assign[idx] = min(list(range(nb_clusters)), key=lambda x: (c_list[x]-d)**2) # find the data's cluster

        for c in range(nb_clusters):
            stuff = [d for idx, d in enumerate(data_) if assign[idx] == c] # calculate the centroids
            if len(stuff) > 0:
                c_list[c] = sum(stuff)/len(stuff)
    return assign






def return_default_mat(im_tensor):
    '''
    im_tensor: mask (torch tensor)
    return default mask and a tensor where bounding box position = 0
    '''
    im_np = im_tensor.numpy()[0, :, :]  # (128,128)
    a = np.where(im_np != 0)
    bbox_int = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0]) # find the boundings of the object

    default_inp = torch.zeros_like(im_tensor) #same size as im_tensor
    idc1 = range(bbox_int[0], bbox_int[2] + 1) #a[1] is column
    idc2 = range(len(idc1))
    for j_, y_ in enumerate(range(bbox_int[1], bbox_int[3] + 1)):
        default_inp[:, j_ + 128 - (-bbox_int[1] + bbox_int[3] + 1), idc2] = im_tensor[:, y_, idc1]

    weight = construct_weight_map(bbox_int)

    return default_inp, weight







def compute_iou(pred, true):
    '''
        pred: 3* num_masks * 128* 128, can apply in photorealistic data
    '''
    nb_objects = pred.size(1)
    pred = torch.sum(pred.view(-1, 3, 128, 128), dim=1) # sum over 3 channels
    true = torch.sum(true.view(-1, 3, 128, 128), dim=1)
    pred[pred.nonzero(as_tuple=True)] = 128 # all nonzero pixels to be 1
    true[true.nonzero(as_tuple=True)] = 128
    total = pred+true
    res = []
    for i in range(total.size(0)): # For each masks
        intersect = torch.sum(total[i].flatten()==256).item()
        union = torch.sum(total[i].flatten()==128).item()
        res.append(intersect/(intersect+union)*1.0) 
    compressed_res = []
    for j in range(0, len(res), nb_objects):
        compressed_res.append(res[j: j+nb_objects])
    assert np.mean(compressed_res) - np.mean(res) <= 0.00001, "%f %f" % (np.mean(compressed_res), np.mean(res))
    return compressed_res, np.mean(compressed_res)


if __name__ == '__main__':
    import time

    for _ in range(1):
        start = time.time()
        read_seg_masks()
        end = time.time()
        print(end-start)
