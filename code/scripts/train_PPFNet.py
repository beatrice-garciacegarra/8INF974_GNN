import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from utils.data_loader import get_ShapeNet
from utils.training import train_PPFNet
from utils.evaluation import evaluate_PN
from utils.transform import transform
from utils.visualizations import scatterplot

# train model on ordinary dataset
traindata, testdata = get_ShapeNet(root = 'data/ShapeNet',transformation = transform(points = 256))

model = train_PPFNet(traindata,n_epochs = 10)
SAVEPATH = 'code/modelsLocal/saved_models/new_PPFNetmodel.pkl'
torch.save(model.state_dict(), SAVEPATH)

#train model on randomly rotated dataset
rotated_traindata, rotated_testdata = get_ShapeNet(root = 'data/ShapeNet',transformation = transform(points = 256,rotate = 180))

rotated_model = train_PPFNet(rotated_traindata,n_epochs = 10)
rotated_SAVEPATH = 'code/modelsLocal/saved_models/new_PPFNetmodel_rotate.pkl'
torch.save(rotated_model.state_dict(), rotated_SAVEPATH)

#evaluate model
print('Acc no rotations on training nor testdata: ' + str(evaluate_PN(model,testdata)))
print('Acc no rotations on training but rotation on testdata: ' + str(evaluate_PN(model,rotated_testdata)))
print('Acc rotations on training and testdata: ' + str(evaluate_PN(rotated_model,rotated_testdata)))