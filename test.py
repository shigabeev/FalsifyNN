import sys
sys.path.insert(0, './squeezeDet/src')

from classify import classify
from populateLibrary import generatePicture



generatePicture((0.5,0.5),'./pics/out/new.png')
#classify()
