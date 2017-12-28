from neural_nets import squeezedet as nn
from modify_primitives.utils import *
from modify_primitives.heatmap import *
from ml_primitives.sampling_primitives import *
from populateLibrary import *
import glob
 
cars = glob.glob("pics/cars/*.png")
roads = glob.glob("pics/roads/*.jpg")

conf = nn.init()


DIM = 2
out = 'pics/out/'

Lib = populateLibrary()

samples = halton_sampling(2, 1)

for road in roads[:1]:
    out_pic_name = out + "tmp.png"
    im = Image.open(road)

    for sample in samples[:1]:
        loc = generatePicture(Lib, [sample[0], sample[1], 1, 1, 1, 1], "tmp.png", i-130, 3)
        confidence = nn.classify(out_pic_name, conf)
        print(confidence)
        if not confidence:
            score = 0
        else:
            try:
                if confidence[0][0][0] == 0:
                    score = int(confidence[0][0][1]*100)
                else:
                    score = 0
            except IndexError:
                break
        print(score)
        col = rgb(0,100,score)
        im = draw_circle(im, loc[0], loc[1], 5, col)
        w.writerow(sample+[score])

    im.save(OUT_PIC_PATH + pic_name)
    f.close()
