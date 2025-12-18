# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 12:05:59 2025

@author: andre
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image 
import math
import numpy as np


#%%

# Function to find the extension to Z2 via symmetrization with respect to -0.5

def symmetrizationZ2 (k, M):
    return min([k%(2*M), (2*M-1-k)%(2*M)])
    
# Function to compute the bilinear interpolation of an image

def bilinear_interpolation(image, 
                           delta = 0.5 # inter-pixel distance of the output image
                           ):
    M = image.shape[0]
    N = image.shape[1]
    
    Mp = math.floor(M/delta)
    Np = math.floor(N/delta)

    u = np.zeros((Mp, Np))
    for m in range(Mp):
        for n in range(Np):
            x = m*delta
            y = n*delta
            xf = math.floor(x)
            yf = math.floor(y)
            u[m,n] = ((x-xf)*(y-yf)*image[symmetrizationZ2(xf+1,M), symmetrizationZ2(yf+1,N)] + 
                (1+xf-x)*(y-yf)*image[symmetrizationZ2(xf,M), symmetrizationZ2(yf+1,N)] +
                (x-xf)*(1+yf-y)*image[symmetrizationZ2(xf+1,M), symmetrizationZ2(yf,N)] +
                (1+xf-x)*(1+yf-y)*image[symmetrizationZ2(xf,M), symmetrizationZ2(yf,N)])
            
    return u

# Function to compute the Gaussian smoothing

def gaussian_smoothing(sigma):
    
    bound = math.ceil(4*sigma)
    kp = np.arange(-bound, bound + 1)
    
    g = np.exp(-kp**2 / (2 * sigma**2))
    g /= g.sum()
    
    return kp, g

    
# Function to apply the Gaussian convolution

def gaussian_convolution(image, sigma):
    
    M, N = image.shape
    indices, kernel = gaussian_smoothing(sigma)
    
    Gu = np.zeros((M, N))
    
    for m in range(M):
        sym_rows = np.array([symmetrizationZ2(m+i, M) for i in indices])
        for n in range(N):             
            sym_cols = np.array([symmetrizationZ2(n+j, N) for j in indices])
            window = image[np.ix_(sym_rows, sym_cols)]
            Gu[m, n] = np.matmul(kernel[None, :], np.matmul(window, kernel[:, None])).item()
                
    return Gu


# Function to compute the digital Gaussian scale-space

def digital_Gaussian_scale_space(image, 
                                 delta = 0.5,
                                 number_octaves = 4,
                                 number_scales = 3, # per octave
                                 sigma_min = 0.8, # blur level in the seed image
                                 delta_min = 0.5, # inter-sample distance in the seed image
                                 sigma_in = 0.5 # asumed blur level in the input image
                                 ):
    
    # Initialize the octaves
    octaves = []
    
    # Compute the first octave
    
    octaves1 = []
    
    # Interpolate the original image
    u = bilinear_interpolation(image, delta_min)
    
    # Gaussian blur
    sigma0 = np.sqrt(sigma_min**2 - sigma_in**2)/delta_min
    v = gaussian_convolution(u, sigma = sigma0)
    octaves1.append(v)
    
    # Compute the other images in the first octave
    for s in range(1,number_scales+3):
        print(s)
        rho = sigma_min/delta_min * np.sqrt(2**(2*s/number_scales)-2**(2*(s-1)/number_scales))
        v = gaussian_convolution(v, sigma = rho)
        octaves1.append(v)
    
    octaves.append(octaves1)
    # Compute the subsequent octaves
    
    M, N = image.shape
    
    for o in range(1,number_octaves):
        
        octaves1 = []
      
        M0 = math.floor(2**(1-o) * M)
        N0 = math.floor(2**(1-o) * N)
        
        # Compute the first image in the octave by subsampling
        v = np.zeros((M0, N0))
        for m in range(M0):
            for n in range(N0):
                v[m,n] = octaves[o-1][number_scales][2*m,2*n]
        octaves1.append(v)
                
        # Compute the other images in the octave
        for s in range(1,number_scales+3):
            print(s)
            rho = sigma_min/delta_min * np.sqrt(2**(2*s/number_scales)-2**(2*(s-1)/number_scales))
            v = gaussian_convolution(v, sigma = rho)
            octaves1.append(v)
            
        octaves.append(octaves1)
        
    return(octaves)


# Function to compute the difference of Gaussian scale-space

def DoG (set_of_octaves):
    
    dog = []
    
    for o in range(len(set_of_octaves)):
        
        dog_inner = []
        
        for s in range(len(set_of_octaves[0])-1):
            dog_inner.append(set_of_octaves[o][s+1] - set_of_octaves[o][s])
            
        dog.append(dog_inner)
        
    return(dog)

# Function to scann for the 3D discrete extrema of the DoG

def extrema_DoG (dog):
    
    extrema = []
    
    for o in range(len(dog)):
        M, N = dog[o][0].shape
        
        for s in range(1,len(dog[0])-1): 
            
            for m in range(1, M-1):
                for n in range(1, N-1):
                    px = dog[o][s][m,n]
                    neighbors = np.concatenate(
                        (dog[o][s-1][range(m-1,m+2),:][:,range(n-1,n+2)].flatten(),
                         np.delete(dog[o][s][range(m-1,m+2),:][:,range(n-1,n+2)].flatten(),4),
                         dog[o][s+1][range(m-1,m+2),:][:,range(n-1,n+2)].flatten()))
                
                    if (px > np.max(neighbors)) or (px < np.min(neighbors)):
                        extrema.append([o,s,m,n])
    
    return extrema

# Function to discard low contrasted candidate keypoints

def filter_extrema_DoG (dog, extrema, threshold_dog = 0.015):
    
    kept_extrema = []
    
    for i in range(len(extrema)):
        if dog[extrema[i][0]][extrema[i][1]][extrema[i][2], extrema[i][3]] >= 0.8 * threshold_dog:
            kept_extrema.append(extrema[i])
            
    return kept_extrema
    
# Function to compute the quadratic interpolation on a discrete DoG sample

def quadratic_interpolation (dog, sample):
    
    current_octave = dog[sample[0]]
    previous_scale = current_octave[sample[1]-1][range(sample[2]-1,sample[2]+2),:][:,range(sample[3]-1,sample[3]+2)]
    current_scale = current_octave[sample[1]][range(sample[2]-1,sample[2]+2),:][:,range(sample[3]-1,sample[3]+2)]
    posterior_scale = current_octave[sample[1]+1][range(sample[2]-1,sample[2]+2),:][:,range(sample[3]-1,sample[3]+2)]
    
    gradient = np.array(([posterior_scale[1,1]-previous_scale[1,1],
                          current_scale[2,1]-current_scale[0,1],
                          current_scale[1,2]-current_scale[1,0]]))/2
    
    Hessian = np.array([[posterior_scale[1,1]+previous_scale[1,1]-2*current_scale[1,1],
                        (posterior_scale[2,1]-posterior_scale[0,1]-previous_scale[2,1]+previous_scale[0,1])/4,
                        (posterior_scale[1,2]-posterior_scale[1,0]-previous_scale[1,2]+previous_scale[1,0])/4],
                       
                        [(posterior_scale[2,1]-posterior_scale[0,1]-previous_scale[2,1]+previous_scale[0,1])/4,
                        current_scale[2,1]+current_scale[0,1]-2*current_scale[1,1],
                        (current_scale[2,2]-current_scale[2,0]-current_scale[0,2]+current_scale[0,0])/4],
                        
                       [(posterior_scale[1,2]-posterior_scale[1,0]-previous_scale[1,2]+previous_scale[1,0])/4,
                        (current_scale[2,2]-current_scale[2,0]-current_scale[0,2]+current_scale[0,0])/4,
                        current_scale[1,2]+current_scale[1,0]-2*current_scale[1,1]]])
    
    invH = np.linalg.inv(Hessian)
    alpha = -np.matmul(invH, gradient[:,None])
    omega = current_scale[1,1] + np.matmul(gradient[None,:], alpha).item()/2
    return alpha, omega


# Function to interpolate keypoints

def keypoints_interpolation (dog, extrema,
                             number_scales = 3, # per octave
                             sigma_min = 0.8, # blur level in the seed image
                             delta_min = 0.5 # inter-sample distance in the seed image
                             ):
    
    candidate_keypoint = []
        
    for i in range(len(extrema)):
        print(i)
        o, s, m, n = extrema[i]
        delta0 = delta_min * 2**(o-1+1)
        repetition = 0
        coordinates = np.zeros(3)
        
        # Compute the local quadratic function
        alpha, omega = quadratic_interpolation(dog, extrema[i])
        if (np.max(np.abs(alpha)) <= 0.5):
            coordinates = np.array([delta0/delta_min * 2**((alpha[0]+s)/number_scales),
                                     delta0*(alpha[1]+m), delta0*(alpha[2]+n)])
            candidate_keypoint.append(np.append(np.concatenate((np.array([o,s,m,n]), coordinates.flatten())), omega))
        
        else:       
            while (np.max(np.abs(alpha)) > 0.5) and (repetition != 5):
            
                repetition +=1
                alpha[alpha < -0.5] = -0.5 + 10**(-15) # To correct the rounding
                alpha[alpha > -0.5] = 0.5
            
                # Compute the corresponding absolute coordinates
                coordinates = np.array([delta0/delta_min * 2**((alpha[0]+s)/number_scales),
                                        delta0*(alpha[1]+m), delta0*(alpha[2]+n)])
            
                # Update the interpolation position
                s = np.min([int(np.round(s+alpha[0]).item()), number_scales])
                m = int(np.round(m+alpha[1]).item())
                n = int(np.round(n+alpha[2]).item())
            
                # Compute the local quadratic function
                alpha, omega = quadratic_interpolation(dog, [o,s,m,n])
    
            if np.max(np.abs(alpha)) < 0.6:
                candidate_keypoint.append(np.append(np.concatenate((np.array([o,s,m,n]), coordinates.flatten())), omega))

    return candidate_keypoint


# Function to discard low contrasted candidate keypoints

def discard_low_contrasted_keypoints(candidate_keypoints, 
                                     threshold_dog = 0.015 # default value for s=3
                                     ):
    candidates = []

    for i in range(len(candidate_keypoints)):
        if np.abs(candidate_keypoints[i][-1]) >= threshold_dog:
            candidates.append(candidate_keypoints[i])
            
    return candidates


# Function to discard candidate keypoints on edges and get the SIFT keypoints

def SIFT_keypoints(dog, candidate_keypoints,
                   threshold_edge = 10):
    
    bound = (threshold_edge+1)**2/threshold_edge
    keypoints = []

    for i in range(len(candidate_keypoints)):
        
        # Compute the 2D Hessian
        sample = [int(candidate_keypoints[i][j]) for j in range(4)]
        
        current_octave = dog[sample[0]]
        previous_scale = current_octave[sample[1]-1][range(sample[2]-1,sample[2]+2),:][:,range(sample[3]-1,sample[3]+2)]
        current_scale = current_octave[sample[1]][range(sample[2]-1,sample[2]+2),:][:,range(sample[3]-1,sample[3]+2)]
        posterior_scale = current_octave[sample[1]+1][range(sample[2]-1,sample[2]+2),:][:,range(sample[3]-1,sample[3]+2)]
        
        Hessian = np.array([[posterior_scale[1,1]+previous_scale[1,1]-2*current_scale[1,1],
                            (posterior_scale[2,1]-posterior_scale[0,1]-previous_scale[2,1]+previous_scale[0,1])/4,
                            (posterior_scale[1,2]-posterior_scale[1,0]-previous_scale[1,2]+previous_scale[1,0])/4],
                           
                            [(posterior_scale[2,1]-posterior_scale[0,1]-previous_scale[2,1]+previous_scale[0,1])/4,
                            current_scale[2,1]+current_scale[0,1]-2*current_scale[1,1],
                            (current_scale[2,2]-current_scale[2,0]-current_scale[0,2]+current_scale[0,0])/4],
                            
                           [(posterior_scale[1,2]-posterior_scale[1,0]-previous_scale[1,2]+previous_scale[1,0])/4,
                            (current_scale[2,2]-current_scale[2,0]-current_scale[0,2]+current_scale[0,0])/4,
                            current_scale[1,2]+current_scale[1,0]-2*current_scale[1,1]]])
        
        # Compute the edgeness
        edgeness = np.trace(Hessian)**2 / np.linalg.det(Hessian)
        
        if edgeness < bound:
            keypoints.append(candidate_keypoints[i])
            
    return keypoints



# Function to compute the gradient at each image of the scale-space

def gradient_octaves (set_of_octaves):
    
    gradient = []
    
    for o in range(len(set_of_octaves)):
        M, N = set_of_octaves[o][0].shape
        gradient1 = []
        
        for s in range(1,(len(set_of_octaves[0])-2)): 
            partialx = np.zeros((M-2,N-2))
            partialy = np.zeros((M-2,N-2))
            for m in range(1,M-1):
                for n in range(1,N-1):
                    partialx[m-1,n-1] = (set_of_octaves[o][s][m+1][n] - set_of_octaves[o][s][m-1][n])/2
                    partialy[m-1,n-1] = (set_of_octaves[o][s][m][n+1] - set_of_octaves[o][s][m][n-1])/2
         
            gradient1.append([partialx, partialy])
            
        gradient.append(gradient1)
        
    return(gradient)



# Function to compute the keypoint reference orientation

def keypoint_orientation(keypoints, gradient,
                         lambda_ori = 1.5, # the patch is 6*lambda_ori*sigma_key wide for
                                           # a keypoint of scale sigma_key
                         number_bins = 36,
                         delta_min = 0.5,
                         threshold = 0.8):
    
    oriented_keypoints = []
    for i in range(len(keypoints)):
        
        # Check if the keypoint is distant enough from the image borders
        o_key = int(keypoints[i][0])
        delta_key = delta_min * 2**(o_key-1+1)     
        height, width = gradient[o_key][0][0].shape 
        height = height*delta_key
        width = width*delta_key
        
        s_key = int(keypoints[i][1])-1
        
        x_key = keypoints[i][5]
        y_key = keypoints[i][6]
        sigma_key = keypoints[i][4]
        
        if ((3*lambda_ori*sigma_key <= x_key <= height-3*lambda_ori*sigma_key) and
            (3*lambda_ori*sigma_key <= y_key <= width-3*lambda_ori*sigma_key)):
            
            # Initialize the orientation histogram
            hist = np.zeros(number_bins)
    
            # Accumulate samples from the normalized patch P_ori
            
            m_min = int(np.round((x_key-3*lambda_ori*sigma_key)/delta_key))
            m_max = int(np.round((x_key+3*lambda_ori*sigma_key)/delta_key))
            n_min = int(np.round((y_key-3*lambda_ori*sigma_key)/delta_key))
            n_max = int(np.round((y_key+3*lambda_ori*sigma_key)/delta_key))            
                        
            for m in range(m_min, m_max+1): 
                for n in range(n_min, n_max+1):
                    
                    # Compute the sample contribution
                    
                    diference = np.array([m*delta_key,n*delta_key])-np.array([x_key,y_key])
                    point_gradient = np.array([gradient[o_key][s_key][0][m-1][n-1],
                                               gradient[o_key][s_key][1][m-1][n-1]])
                    contribution = (np.exp(-np.linalg.norm(diference)**2/(2*(lambda_ori*sigma_key)**2))
                                    * np.linalg.norm(point_gradient))
                    
                    # Compute the arctang mod 2pi
                    arc_tan = (np.arctan2(point_gradient[0], point_gradient[1])+2*np.pi) % (2*np.pi)
                    
                    # Compute the corresponding bin index
                    bin_ori = int(np.round(number_bins/(2*np.pi) * arc_tan))

                    # Update the histogram
                    hist[bin_ori-1] = hist[bin_ori-1] + contribution
                    
            # Smooth the histogram
            
            kernel = np.ones(3) / 3
            kernel_padded = np.zeros(len(hist))
            kernel_padded[:3] = kernel
            
            # Apply six times
            repetition = 0
            
            while repetition != 6:
                hist = np.fft.ifft(np.fft.fft(hist) * np.fft.fft(kernel_padded)).real
                repetition +=1
                
            # Extract the reference orientations
            
            for k in range(1, number_bins+1):
                k_minus = (k-1) % number_bins 
                k_plus = (k+1) % number_bins 
                
                if hist[k-1] > np.max([hist[k_minus-1], hist[k_plus-1], threshold*np.max(hist)]): # -1 is for the starting in 0
                    
                    # Compute the reference orientation 
                    theta = 2*np.pi*k / number_bins
                    theta_key = theta + (np.pi/number_bins * 
                                         ((hist[k_minus-1]-hist[k_plus-1])/
                                          (hist[k_minus-1]-2*hist[k-1]+hist[k_plus-1])))
                    
                    oriented_keypoints.append(np.append(keypoints[i], theta_key))
                    
    return oriented_keypoints


# Function to compute the keypoint descriptor

def keypoint_descriptor(oriented_keypoints, gradient,
                         lambda_ori = 1.5, # the patch is 6*lambda_ori*sigma_key wide for
                                           # a keypoint of scale sigma_key
                         number_hist = 4, # the descriptor is an array of number_hist x number_hist
                                          # orientation histograms
                         number_ori = 8, # number of bins in the orientation histograms
                         lambda_descr = 6, # the Gaussian window has a standard deviation of
                                          # lamda_descr * sigma_key
                         delta_min = 0.5):
                                              
    features = []
    
    # Compute all ^x_i, ^y_j
    associated_positions = [(pos-(1+number_hist)/2) * 2*lambda_descr/number_hist for pos in range(number_hist)]
    
    # Compute ^theta_k
    hist_center = [(2*np.pi*(ori-1+1)/number_ori+2*np.pi) % (2*np.pi) for ori in range(number_ori)]
    # hist_center = [2*np.pi*(ori-1)/number_ori for ori in range(number_ori)]
    
    for i in range(len(oriented_keypoints)):
        print(i)
        
        # Check if the keypoint is distant enough from the image borders
        o_key = int(oriented_keypoints[i][0])
        delta_key = delta_min * 2**(o_key-1+1)     
        height, width = gradient[o_key][0][0].shape 
        height = height*delta_key
        width = width*delta_key
        
        s_key = int(oriented_keypoints[i][1])-1
        
        x_key = oriented_keypoints[i][5]
        y_key = oriented_keypoints[i][6]
        sigma_key = oriented_keypoints[i][4]
        theta_key = oriented_keypoints[i][8]
        
        if ((np.sqrt(2)*lambda_descr*sigma_key <= x_key <= height-np.sqrt(2)*lambda_descr*sigma_key) and
            (np.sqrt(2)*lambda_descr*sigma_key <= y_key <= width-np.sqrt(2)*lambda_descr*sigma_key)):
            
            # Initialize the array of weighted histograms
            
            histograms = [[np.zeros((number_ori)) for _ in range(number_hist)] for _ in range(number_hist)]
            
            # Accumulate samples from the normalized patch P_descr
                        
            coeff = np.sqrt(2)*lambda_descr*sigma_key
            m_min = int(np.round((x_key-coeff)/delta_key))
            m_max = int(np.round((x_key+coeff)/delta_key))
            n_min = int(np.round((y_key-coeff)/delta_key))
            n_max = int(np.round((y_key+coeff)/delta_key))            
                        
            for m in range(m_min, m_max+1): 
                for n in range(n_min, n_max+1):
                                
                    # Compute the normalized coordinates
                                
                    x_norm = (((m*delta_key-x_key)*np.cos(theta_key) +
                               (n*delta_key-y_key)*np.sin(theta_key)) / sigma_key)
                    y_norm = ((-(m*delta_key-x_key)*np.sin(theta_key) +
                              (n*delta_key-y_key)*np.cos(theta_key)) / sigma_key)
                    
                    # Verify if the sample is inside the normalized patch
                                
                    if (np.max([np.abs(x_norm), np.abs(y_norm)]) < 
                            lambda_descr*(number_hist+1)/number_hist):
                                    
                        # Compute the normalized gradient orientation
                                    
                        point_gradient = np.array([gradient[o_key][s_key][0][m-1][n-1],
                                               gradient[o_key][s_key][1][m-1][n-1]])
                        theta_norm = (np.arctan2(point_gradient[0], point_gradient[1])
                                  -theta_key+2*np.pi) % (2*np.pi)
                                
                        # Compute the total contribution of the sampla*n_e
                                    
                        diference = np.array([m*delta_key,n*delta_key])-np.array([x_key,y_key])
                        contribution = (np.exp(-np.linalg.norm(diference)**2/(2*(lambda_descr*sigma_key)**2))
                                        * np.linalg.norm(point_gradient))

                        # Update the nearest histograms and the nearest bins
                                    
                        for a in range(number_hist):
                            if np.abs(associated_positions[a]-x_norm) <= 2*lambda_descr/number_hist:
                                        
                                for b in range(number_hist):
                                    if np.abs(associated_positions[b]-y_norm) <= 2*lambda_descr/number_hist:
                                                    
                                        for k in range(number_ori):
                                            if (np.abs((hist_center[k]-theta_norm+2*np.pi) % (2*np.pi)) <
                                                2*np.pi/number_ori):
                                                            
                                                histograms[a][b][k] += ((1-number_hist/(2*lambda_descr)*
                                                                      np.abs(x_norm-associated_positions[a]))*
                                                                     (1-number_hist/(2*lambda_descr)*
                                                                      np.abs(y_norm-associated_positions[b]))*
                                                                     (1-number_ori/(2*np.pi)*
                                                                      np.abs((theta_norm-hist_center[k]+2*np.pi) % 
                                                                             (2*np.pi))) * contribution)

            # Build the feature vector from the array of weighted histograms
        
            f = np.zeros(number_hist*number_hist*number_ori)
            for a in range(number_hist): 
                for b in range(number_hist):
                    for k in range(number_ori): 
                        f[a*number_hist*number_ori+b*number_ori+k] = histograms[a][b][k]
        
        
            # Renormalize
        
            f_norm = np.linalg.norm(f)
            f_normalized = [np.min([f[l], 0.2*f_norm]) for l in range(len(f))]
        
            # Quantize to 8 bit integers
        
            f_normalized_norm = np.linalg.norm(f_normalized)
            f_integer = [np.min([np.floor(512*f_normalized[l]/f_normalized_norm), 255]) for l in range(len(f_normalized))]
        
            features.append([x_key, y_key, sigma_key, theta_key, f_integer])
    
    return features


# Function for matching points

def matching(keydes1, keydes2, 
             threshold_matching = 0.6 # relative threshold
             ):
    
    matches = []
    
    for i in range(len(keydes1)):
        
        # Find all distances to the descriptors in the second set
        distances = [np.linalg.norm(np.array(keydes1[i][-1])-np.array(keydes2[j][-1])) for j in range(len(keydes2))]
        
        # Fint the two nearest descriptors
        nearest_descriptors = np.sort(distances)[:2]
        
        # Select pair satisfying a relative threshold
        
        if nearest_descriptors[0] < threshold_matching * nearest_descriptors[1]:
            
            matches.append([keydes1[i], keydes2[np.argmin(distances)]])
    
    return matches


# Funtion to find the SIFT keypoints and descriptors of an image

def SIFT_keypoints_descriptors(image, 
                               delta = 0.5,
                               number_octaves = 4,
                               number_scales = 3, # per octave
                               sigma_min = 0.8, # blur level in the seed image
                               delta_min = 0.5, # inter-sample distance in the seed image
                               sigma_in = 0.5 # asumed blur level in the input image
                               ):
    
    # Compute the Gaussian scale-space
    set_of_octaves = digital_Gaussian_scale_space(image)
    
    # Compute the Difference of Gaussians
    dog = DoG(set_of_octaves)
    
    # Find 3D discrete extrema of DoG
    extrema = extrema_DoG(dog)
    
    # Discard low contrasted candidate keypoints
    filtered_extrema = filter_extrema_DoG(dog, extrema)
       
    # Refine candidate keypoints location with sub-pixel precision
    interpolated_extrema = keypoints_interpolation(dog, filtered_extrema)
    
    # Filter unstable keypoints due to noise
    candidate_keypoints = discard_low_contrasted_keypoints(interpolated_extrema)
    
    # Filter unstable keypoints lying on edges
    keypoints = SIFT_keypoints(dog, candidate_keypoints)
    
    # Assign a reference orientation to each point    
    octaves_gradient = gradient_octaves(set_of_octaves)
    oriented_keypoints = keypoint_orientation(keypoints, octaves_gradient)
    
    # Build the keypoints descriptor
    keydes = keypoint_descriptor(oriented_keypoints, octaves_gradient)
    
    return keydes


# Function to plot matching

def plot_matching(image1, image2, matching_points):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Points (x, y) in each image to connect
    points_img1 = [(np.round(matching_points[i][0][0]), np.round(matching_points[i][0][1]))
      for i in range(len(matching_points))]
    points_img2 = [(np.round(matching_points[i][1][0]), np.round(matching_points[i][1][1]))
      for i in range(len(matching_points))]


    # Show images
    ax1.imshow(image1, cmap='gray')
    ax1.axis('off')

    ax2.imshow(image2, cmap='gray')
    ax2.axis('off')

    # Plot points on each image
    x1, y1 = zip(*points_img1)
    ax1.plot(y1, x1, 'ro', markersize=8)

    plt.show()

    x2, y2 = zip(*points_img2)
    ax2.plot(y2, x2, 'ro', markersize=8)

    plt.show()

    # Connect corresponding points between the two images
    for (x1_p, y1_p), (x2_p, y2_p) in zip(points_img1, points_img2):
        con = patches.ConnectionPatch(
            xyA=(y2_p, x2_p), coordsA=ax2.transData,
            xyB=(y1_p, x1_p), coordsB=ax1.transData,
            arrowstyle='-', color='blue', linewidth=2)
        fig.add_artist(con)

    plt.tight_layout()
    plt.show()

def plot_SAR_matching(image1, image2, matching_points, maximum=0.25):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Points (x, y) in each image to connect
    points_img1 = [(np.round(matching_points[i][0][0]), np.round(matching_points[i][0][1]))
      for i in range(len(matching_points))]
    points_img2 = [(np.round(matching_points[i][1][0]), np.round(matching_points[i][1][1]))
      for i in range(len(matching_points))]


    # Show images
    ax1.imshow(image1, cmap='gray', vmin=0, vmax=maximum)
    ax1.axis('off')

    ax2.imshow(image2, cmap='gray', vmin=0, vmax=maximum)
    ax2.axis('off')

    # Plot points on each image
    x1, y1 = zip(*points_img1)
    ax1.plot(y1, x1, 'ro', markersize=8)

    x2, y2 = zip(*points_img2)
    ax2.plot(y2, x2, 'ro', markersize=8)

    # Connect corresponding points between the two images
    for (x1_p, y1_p), (x2_p, y2_p) in zip(points_img1, points_img2):
        con = patches.ConnectionPatch(
            xyA=(y2_p, x2_p), coordsA=ax2.transData,
            xyB=(y1_p, x1_p), coordsB=ax1.transData,
            arrowstyle='-', color='blue', linewidth=1)
        fig.add_artist(con)

    plt.tight_layout()
    plt.show()
