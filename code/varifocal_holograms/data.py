import glob
import torch
from odak import np
from odak.learn.tools import load_image,save_image
from odak.tools import check_directory,load_dictionary

def save(field,filename='output.png',directory='./',save_type='image',cmin=0,cmax=1.0):
    check_directory(directory)
    fn = '{}/{}'.format(directory,filename)
    if save_type == 'image':
         field_save = field*255
    elif save_type == 'phase':
        field_save = field%(2*np.pi)/(2*np.pi)*255
    save_image(fn,field_save,cmin=cmin,cmax=cmax*255)

def load(fn,device):
    target = load_image(fn).to(device).double()
    if len(target.shape) > 2:
        if target.shape[2] > 2:
            target = target[:,:,0:3]
    if target.max() > 1.: 
        divider = 255.
        target  = target/divider
#    for i in range(2):
#        target = (torch.exp(target)-torch.exp(torch.tensor(0)))/(torch.exp(torch.tensor(1.0))-torch.exp(torch.tensor(0)))
    new_target      = torch.zeros(1,target.shape[2],target.shape[0],target.shape[1]).to(device)
    new_target[0,0] = target[:,:,0]
    new_target[0,1] = target[:,:,1]
    new_target[0,2] = target[:,:,2]
    return new_target.float()

class DatasetFromFolder():
    def __init__(self,input_directory,device,keys=['.png','.json']):
        self.device               = device
        self.keys                 = keys
        self.input_directory      = input_directory
        self.input_filenames      = sorted(glob.glob(input_directory+'/**/*{}'.format(self.keys[0]),recursive=True))
        self.input_meta_filenames = sorted(glob.glob(input_directory+'/**/*{}'.format(self.keys[1]),recursive=True))

    def __getitem__(self, index):
        input_image = load(self.input_filenames[index],self.device)
        meta_data   = load_dictionary(self.input_meta_filenames[index])
        return input_image,meta_data

    def __len__(self):
        return len(self.input_filenames)

