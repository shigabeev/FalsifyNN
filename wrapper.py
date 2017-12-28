from modify_primitives.utils import *
from modify_primitives.clustering_bo import *
from populateLibrary import *
from neural_nets import squeezedet as nn
from modify_primitives import library
from modify_primitives.components import ImageFile

Lib = library()

def add_element(path, type, desc, data=None, Lib = Lib):
    # data = (vanishing point, mix_x, max_x, lanes), selected for road only
    
    # Examples:
    # Lib.addRoad(ImageFile(Image.open("./pics/roads/desert.jpg"), "Desert Road"), coord(800, 540), coord(100, 950), coord(1500,950), [coord(800,950)])
    # Lib.addCar(ImageFile(Image.open("./pics/cars/bmw_rear.png"), "BMW"))
    # Note that ImageFile returns 2 values
    if type == 'road':    # (data, description, vanishing point, mix_x, max_x, lanes)
        Lib.addRoad(ImageFile(Image.open(path), desc) ,data[0], data[1], data[2], data[3])
        return Lib
    elif type == 'car':      # (data, description)
        Lib.addCar(ImageFile(Image.open(path), desc))
        return Lib

def open_roads_csv(path):
    import pandas as pd
    csv = pd.open_csv(path)
    for road in csv.iterrows:
        add_element(road[0], 'road', road[1], road[2:])


# params[0]: car x pos
# params[1]: car y pos
# scene[0]: road
# scene[1]: car

def wrapper(params):

    params = params[0]

    out_pic_name = "tmp.png"                                        #########################
    real_box = generatePicture(Lib, [params[0], params[1], 1, 1, 1, 1], out_pic_name, 0, 0)
    (boxes,probs,cats) = nn.classify(out_pic_name, conf)            #########################

    # extract max prob box
    max_prob = 0

    for box, prob, cat in zip(boxes, probs, cats):
        if (prob > max_prob) and (cat == 0):
            max_prob = prob
            max_prob_box = box
            max_prob_cat = cat

    return max_prob





add_element('pics/roads/desert.jpg', type='road', desc="Desert Road", data=[coord(800, 540), coord(100, 950), coord(1500,950), coord(800,950)], Lib = Lib)
add_element('pics/cars/peugeot_kitti_full_HD.png', type='car', desc ='fullhd', Lib = Lib)

conf = nn.init()

opt = bo_class(input_dim=2)
opt.init_BO(f=wrapper)

for _ in range(1):
    opt.run_BO(max_iter=1)
    print opt.bo.suggested_sample
