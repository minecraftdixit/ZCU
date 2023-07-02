from torchvision import datasets, models, transforms
import os
import sys
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

#from common import *


DIVIDER = '-----------------------------------------'

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((120,120)),
        transforms.RandomResizedCrop(120),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.45, 0.46, 0.41], [0.231, 0.226, 0.228])
    ]),
    'val': transforms.Compose([
        transforms.Resize(120),
        transforms.CenterCrop(120),
        transforms.ToTensor(),
        transforms.Normalize([0.45, 0.46, 0.41], [0.231, 0.226, 0.228])
    ]),
}



val_data = 'content/asl_alphabet_train/asl_alphabet_train'
train_data = 'content/asl_alphabet_train/asl_alphabet_train'
image_datasets = {'train':datasets.ImageFolder(train_data,data_transforms['train']),
                   'val': datasets.ImageFolder(val_data,data_transforms['val'])}
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32,
                                             shuffle=True),
               'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=32,
                                             shuffle=True)}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

val_dataloader = dataloaders['val']

  
def test(model, device, val_dataloader):
    '''
    test the model
    '''
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
      for data in dataloaders['val']:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
      
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %') 

    return




def quantize(build_dir,quant_mode,batchsize):

  #dset_dir = build_dir + '/dataset'
  #float_model = build_dir + '/float_model'
  quant_model = build_dir +'./quant_model'


  # use GPU if available   
  #if (torch.cuda.device_count() > 0):
  #  print('You have',torch.cuda.device_count(),'CUDA devices available')
   # for i in range(torch.cuda.device_count()):
   #   print(' Device',str(i),': ',torch.cuda.get_device_name(i))
   # print('Selecting device 0..')
   # device = torch.device('cpu')
 # else:
  #  print('No CUDA devices available..selecting CPU')
  device = torch.device('cpu')

  # load trained model
  model = ConvNet().to(device)
  model.load_state_dict(torch.load('cnn1.pth',map_location ='cpu'))

  # force to merge BN with CONV for better quantization accuracy
  optimize = 1

  # override batchsize if in test mode
  if (quant_mode=='test'):
    batchsize = 1
  
  rand_in = torch.randn([batchsize, 3, 120, 120])
  quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
  quantized_model = quantizer.quant_model



  
  
  
  
  
  
  
  
  
  
  
 # test_dataset = torchvision.datasets.MNIST(dset_dir,
  #                                          train=False, 
  #                                          download=True,
  #                                          transform=test_transform)
#
  #test_loader = torch.utils.data.DataLoader(test_dataset,
   #                                         batch_size=batchsize, 
  #                                          shuffle=False)

  # evaluate 
  test(quantized_model, device, val_dataloader)


  # export config
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if quant_mode == 'test':
    quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
  
  return

#ConvNet implementation
class ConvNet(nn.Module): 
  def __init__(self):
    super(ConvNet,self).__init__()
    self.conv1 = nn.Conv2d(3,4,5)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(4,8,5)
    self.conv3 = nn.Conv2d(8,16,3)
    self.conv4 = nn.Conv2d(16,32,5)
    self.conv5 = nn.Conv2d(32,64,3)
    self.fc1 = nn.Linear(23*23*64,500)
    self.fc2 = nn.Linear(500,250)
    self.fc3 = nn.Linear(250,100)
    self.fc4 = nn.Linear(100,29)
   
  def forward(self,x):
    x = F.relu(self.conv1(x))
    x = self.pool(F.relu(self.conv2(x)))
    x = F.relu(self.conv3(x))
    x = self.pool(F.relu(self.conv4(x)))
    x = F.relu(self.conv5(x))
    x = x.view(-1,23*23*64) 
    # to flatten the image
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x) 
    return x

def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
  ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
  ap.add_argument('-b',  '--batchsize',  type=int, default=100,        help='Testing batchsize - must be an integer. Default is 100')
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('PyTorch version : ',torch.__version__)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print ('--build_dir    : ',args.build_dir)
  print ('--quant_mode   : ',args.quant_mode)
  print ('--batchsize    : ',args.batchsize)
  print(DIVIDER)

  quantize(args.build_dir,args.quant_mode,args.batchsize)

  return



if __name__ == '__main__':
    run_main()

