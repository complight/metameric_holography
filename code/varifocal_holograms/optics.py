import sys
import torch
import torch.nn as nn
import torch.optim as optim
import os
from odak.learn.wave import wavenumber,generate_complex_field,calculate_amplitude,propagate_beam,linear_grating,calculate_phase
from odak.learn.tools import zero_pad,crop_center
from odak.tools import check_directory
from data import DatasetFromFolder,save
from tqdm import tqdm

import odak.learn
from odak.learn.perception import MetamericLoss, BlurLoss, MetamerMSELoss

def prepare(settings):
    if settings["general"]["cuda"]:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(settings["general"]["gpu no"])
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    torch.random.seed()
    kernel_float    = torch.load(settings["general"]["kernel filename"]).to(device)
    kernel          = generate_complex_field(kernel_float[0,0],kernel_float[0,1]).detach()
    kernel          = zero_pad(kernel)
    dataset         = DatasetFromFolder(
                                        settings["dataset"]["input directory"],
                                        device,
                                        keys=settings["dataset"]["keys"]
                                       )
    criterion       = [
                       MetamericLoss(
                                     real_viewing_distance=settings["metameric"]["real viewing distance"],
                                     real_image_width=settings["metameric"]["real image width"],
                                     use_fullres_l0 = False,
                                     use_l2_foveal_loss = True
                                    ).to(device),
                       nn.MSELoss().to(device),
                       BlurLoss(blur_source=False).to(device),
                       BlurLoss(blur_source=True).to(device),
                       MetamerMSELoss(
                                      real_viewing_distance=settings["metameric"]["real viewing distance"],
                                      real_image_width=settings["metameric"]["real image width"]
                                     ).to(device)
                      ]
    checker_complex = linear_grating(
                                     settings["slm"]["resolution"][0],
                                     settings["slm"]["resolution"][1],
                                     axis='y'
                                    ).to(device)
    checker         = calculate_phase(checker_complex)
    return kernel,checker,dataset,criterion,device 

def evaluate(image,target,criterion,header=None,w=[1.,0.,0.,0.,0.]):
    gaze            = header["gaze location"]
    metameric_loss  = 0
    l2_loss         = 0
    blur_m_loss     = 0 
    blur_l_loss     = 0
    l2_metamer_loss = 0
    if w[0] != 0:
        metameric_loss = w[0]*criterion[0](image,target,gaze=(gaze[0],gaze[1]))
    if w[1] != 0:
        l2_loss = w[1]*criterion[1](image,target)
    if w[2] != 0:
        blur_m_loss = w[2]*criterion[2](image,target,gaze=(gaze[0],gaze[1]))
    if w[3] != 0:
        blur_l_loss = w[3]*criterion[3](image,target,gaze=(gaze[0],gaze[1]))
    if w[4] != 0:
        l2_metamer_loss = w[4]*criterion[4](image,target,gaze=(gaze[0],gaze[1]))
    loss = metameric_loss+l2_loss+blur_m_loss+blur_l_loss+l2_metamer_loss
    return loss

def optimize(settings,kernel,dataset,criterion,device,grating=None,wavelength_id=0):
    pixel_pitch          = settings["slm"]["pixel pitch"]
    resolution           = settings["slm"]["resolution"]
    wavelengths          = settings["beam"]["wavelengths"]
    directory            = settings["general"]["output directory"]
    propagation_type     = settings["general"]["propagation type"]
    loss_weights         = settings["general"]["loss weights"]
    learning_rate        = settings["general"]["learning rate"]
    n_iterations         = settings["general"]["iterations"]
    m                    = settings["general"]["region of interest"]
    hologram_no          = settings["general"]["hologram no"]
    multiplier           = settings["general"]["multiplier"]
    ones                 = torch.ones(resolution[0],resolution[1],requires_grad=False).to(device)
    mask                 = torch.zeros(3,resolution[0],resolution[1],requires_grad=False).to(device)
    mask[
         :,
         int(resolution[0]*m[0]):int(resolution[0]*m[1]),
         int(resolution[1]*m[0]):int(resolution[1]*m[1])
        ]                = 1
    if type(kernel) != type(None):
        kernel.requires_grad = False
    t_target             = tqdm(range(dataset.__len__()),leave=False)
    for target_id in t_target: 
        reconstruction = torch.zeros((ones.shape[0],ones.shape[1],3),requires_grad=False).to(device)
        temporal_image = torch.zeros(reconstruction.shape,requires_grad=False).to(device)
        for h in range(hologram_no):
            if h ==0:
                input_phase = torch.rand(3,resolution[0],resolution[1]).detach().to(device).requires_grad_()
                n_iterations  = settings["general"]["iterations"]
            else:
                input_phase   = input_phase.detach().clone().requires_grad_()
                n_iterations  = int(settings["general"]["iterations"]*0.5)
            optimizer          = optim.Adam([{'params': [input_phase]}],lr=learning_rate)
            target_orig,header = dataset.__getitem__(target_id)
            if type(header) != type(None):
                distance      = settings["image"]["location"][2]+header["distance"]*0.001
            else:
                distance      = settings["image"]["location"][2]
            t_optim            = tqdm(range(n_iterations),leave=False)
            noise              = torch.rand(target_orig.shape).to(device)*0.01
            target             = target_orig.detach().clone()+noise
            target             = target*multiplier*mask
            description        = "Hologram no:{}, target no:{}".format(h,target_id)
            t_target.set_description(description)
            for n in t_optim:
                image_rgb = torch.zeros(target.shape).to(device) 
                optimizer.zero_grad()
                for wavelength_id,wavelength in enumerate(wavelengths):
                    field      = a_single_step(
                                               ones.detach().clone(),
                                               input_phase[wavelength_id],
                                               kernel,
                                               distance,
                                               wavelength,
                                               pixel_pitch,
                                               propagation_type
                                              )
                    image_rgb[
                              0,
                              wavelength_id,
                             ] = calculate_amplitude(field)**2
                loss = evaluate(image_rgb*mask,target,criterion,header=header,w=loss_weights)
                loss.backward(retain_graph=True)
                optimizer.step()
                description = "Iteration:{}, Loss:{:.4f}".format(n,loss.item()) 
                t_optim.set_description(description)
            print("Final loss %f" % loss.item())
            image_rgb              = image_rgb.detach().clone()*mask
            reconstruction[:,:,0]  = image_rgb[0,0].detach().clone()
            reconstruction[:,:,1]  = image_rgb[0,1].detach().clone()
            reconstruction[:,:,2]  = image_rgb[0,2].detach().clone() 
            temporal_image[:,:,0] += image_rgb[0,0].detach().clone()
            temporal_image[:,:,1] += image_rgb[0,1].detach().clone()
            temporal_image[:,:,2] += image_rgb[0,2].detach().clone()
            save_results(
                         input_phase.detach().clone()*mask,
                         reconstruction.detach().clone(),
                         temporal_image.detach().clone(),
                         target[0].detach().clone(),
                         grating.clone()*mask[0],
                         directory,
                         h,
                         target_id+1,
                         h,
                         multiplier
                        )
    return True

def save_results(phase,reconstruction,temporal_reconstruction,target,grating,directory='./output',temporal_id=0,result_id=0,number_of_holograms=15,multiplier=1.0):
    check_directory(directory)
    save(reconstruction,filename='reconstruction_{0:04d}_{1:04d}.png'.format(result_id,temporal_id),directory=directory,save_type='image',cmax=multiplier)
    save(temporal_reconstruction,filename='total_reconstruction_{0:04d}.png'.format(result_id),directory=directory,save_type='image',cmax=number_of_holograms*multiplier)
#    torch.save(temporal_reconstruction,'{0}/total_reconstruction_{1:04d}.pt'.format(directory,result_id))
    new_target        = torch.zeros(reconstruction.shape)
    new_target[:,:,0] = target[0]
    new_target[:,:,1] = target[1]
    new_target[:,:,2] = target[2]
    save(new_target,filename='target_{0:04d}_{1:04d}.png'.format(result_id,temporal_id),directory=directory,save_type='image')
    for wavelength_id in range(3):
        save(phase[wavelength_id],filename='hologram_phase_{0}_{1:04d}_{2:04d}.png'.format(wavelength_id,result_id,temporal_id),directory=directory,save_type='phase')
        save(phase[wavelength_id]+grating,filename='hologram_phase_grating_{0}_{1:04d}_{2:04d}.png'.format(wavelength_id,result_id,temporal_id),directory=directory,save_type='phase')

def a_single_step(hologram_amplitude,hologram_phase,kernel,distance,wavelength,pixel_pitch,propagation_type):
    field        = generate_complex_field(hologram_amplitude,hologram_phase) 
    k            = wavenumber(wavelength)
    field_padded = zero_pad(field) 
    if propagation_type != 'custom':
        final_field_padded = propagate_beam(field_padded,k,distance,pixel_pitch,wavelength,propagation_type)
    elif propagation_type == 'custom':
        final_field_padded = propagate_beam(field_padded,None,None,None,None,propagation_type='custom',kernel=kernel)
    final_field        = crop_center(final_field_padded)
    return final_field

def start(settings):
    kernel,checker,dataset,criterion,device = prepare(settings)
    optimize(settings,kernel,dataset,criterion,device,checker)
    return True
