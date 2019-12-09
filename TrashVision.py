import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

from PIL import Image

import sys, os, glob

def read_image(path):
    try:
        image = Image.open(path)
    except:
        print("File not found. Please try again.")
        return None
    transform = transforms.Compose([transforms.Resize([256,256]), transforms.ToTensor()])
    t = transform(image)
    t.unsqueeze_(0)
    return t

def build_models():
    device = torch.device('cpu')
    print('Using {}'.format(device))

    organic_model = torch.load('OrganicModel.pt', map_location=torch.device('cpu'))
    trash_model = models.densenet161(pretrained=False)
    organic_model.to(device)
    trash_model.to(device)
    dict = torch.load('modelThreeDec2.pt', map_location=torch.device('cpu'))
    trash_model.load_state_dict(dict)

    organic_model.eval()
    trash_model.eval()

    return organic_model, trash_model

def run_models(organic_model, trash_model, image):
    with torch.no_grad():
        organic_output = organic_model(image)
        org_out = F.softmax(organic_output.data, dim=1)
        trash_output = trash_model(image)
        trash_out = F.softmax(trash_output.data, dim=1)

        orgscore, org_pred = torch.max(org_out, 1)
        trashscore, trash_pred = torch.max(trash_out, 1)

        score = float(orgscore)
        trashscore = float(trashscore)
        if org_pred == 0 and orgscore >= 0.8:
            print('\tI think this is food waste (with certainty {0:.2f}%), you should compost it.'.format(score*100))
        elif trash_pred == 0:
            print('\tI think this is cardboard (with certainty {0:.2f}%), you should recycle or compost it.'.format(trashscore*100))
        elif trash_pred == 1:
            print('\tI think this is glass (with certainty {0:.2f}%), you should recycle it.'.format(trashscore*100))
        elif trash_pred == 2:
            print('\tI think this is metal (with certainty {0:.2f}%), you should recycle.'.format(trashscore*100))
        elif trash_pred == 3:
            print('\tI think this is paper (with certainty {0:.2f}%), you should compost or recycle it.'.format(trashscore*100))
        elif trash_pred == 4:
            print('\tI think this is plastic (with certainty {0:.2f}%), you should recycle.'.format(trashscore*100))
        elif trash_pred == 5:
            print('\tI think this is other waste (with certainty {0:.2f}%), you should incinerate it/throw it in the garbage.'.format(trashscore*100))

def main():
    print('Loading models...')
    organic_model, trash_model = build_models()
    imfile = input('Enter the path to an image (or an image folder): ')
    while imfile is not '':
        if os.path.exists(imfile):
            if os.path.isdir(imfile) or os.path.isdir(imfile+'/'):
                if imfile[-1] is not '/':
                    imfile = imfile+'/'
                files = glob.glob(imfile+'*.jpg')
            else:
                files = [imfile]
            files.sort()
            for f in files:
                print('For: '+f)
                im = read_image(f)
                if im is not None:
                    run_models(organic_model, trash_model, im)

            imfile = input('Enter the path to an image (or an image folder): ')

if __name__ == '__main__':
    main()
