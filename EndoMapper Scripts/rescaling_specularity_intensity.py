import os
import cv2
import numpy as np
import natsort
import matplotlib.pyplot as plt

def scaling_specularities(image, mask):
    '''Function which inpaints the image with high intensity white pixels according to the white pixels from the mask
    
    Input:

    Image: The image frame which is being augmented with high intensity white pixels.
    Mask: The mask which determines which pixels in the Image Frame are to be augmented.

    Output:
    
    Augmented Image: The image frame after being augmented with high intensity white pixels/specularities. 
    '''
    #pixel which should be augmented, augment the image with the maximum pixel intensity. Applied to each channel in the rgb channels
    #separately and concatenate together.

    rescaled_spec_channel_b = np.where(mask > 200, 255, image[:,:,0])
    rescaled_spec_channel_g = np.where(mask > 200, 255, image[:,:,1])
    rescaled_spec_channel_r =np.where(mask > 200, 255, image[:,:,2])

    #Stack the masked image channels into BGR 3D array to be written/saved.
    rescaled_spec_bgr = np.dstack((rescaled_spec_channel_b, rescaled_spec_channel_g, rescaled_spec_channel_r))

    return rescaled_spec_bgr

def mask_extraction(image):
    '''Function which determines the mask of pixels which can be inpainted with white pixel values (255) to crudely simulate specularities from real colonoscopy video
    with high intensities and sharp gradients (as opposed to the smoother gradient specularities in the EndoMapper dataset).

    Input: 

    Image: The image frame which is being augmented with high intensity white pixels.

    Output:

    Mask: The mask for the image which is implemented for determining which pixels can have their saturation altered.  
    '''

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Since not all of the specularities have sharp intensity gradients, and the inpainting mask is formed from the intersection between high intensity (white) pixels and
    #high image gradients, a Gaussian blur is not that necessary. 

    #sobel_edges = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=1)/255
    #Output of applying the Sobel filter is rescaled onto the [0, 1] pixel range.

    #sobel_binarised =  np.where(sobel_edges > 0.05, 1, 0)
    # Thresholding the pixels which correspond to edge pixels under the use of a primitive edge detector. Parameters are tuned manually to ensure a sufficient quantity of
    # edges are detected, such that applying the second mask corresponding to high intensity pixels isolates a sizeable quantity of the specularity pixels. 

    #Blurring to smooth out the shape of the mask, i.e. less pixellated/blocky white pixels.

    unblurred_sobel = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=1)/255
    high_gradient =  np.where((unblurred_sobel) > 0.05, 1, 0)#0.2, 1, 0)


    #high_canny = np.where(lapl > 10, 1, 0)
    mask1 = cv2.GaussianBlur(np.array(255*high_gradient).astype(np.uint8), (5,5), 0)
 
 
    #Extracting a mask of the pixels with very high intensity pixel values. Blurring to also smooth out the shape of the mask pixels (less pixellated)
    high_intensity_pixel_mask = cv2.GaussianBlur(np.where(grayscale > 235, 255, 0).astype(np.uint8), (5,5), 0)  #cv2.GaussianBlur(np.where(grayscale > 240, 255, 0).astype(np.uint8), (5,5), 0)

    #Computing a mask from the intersection of the two masks, then dilating the  "white" pixels in order to obtain adequately sized inpainted specularities.
    #The number of iterations for dilating the intersected mask is also determined manually by inspection. 
    mask_intersection = high_intensity_pixel_mask * mask1 #sobel_mask
    kernel = np.ones((3,3), np.uint8)
    mask2 = cv2.dilate(mask_intersection.astype(np.uint8), kernel, iterations = 1)
    #mask2_smoothed = cv2.GaussianBlur(mask2.astype(np.uint8), (1,1), 0)
    #plt.imshow(high_intensity_pixel_mask) #(cv2.imshow('highintensity', high_intensity_pixel_mask)
    #cv2.imshow('highgrad', sobel_mask)
    #cv2.waitKey(0)
    return mask2#_smoothed 

def main():
    endomapper_dir = os.path.join(os.path.split(os.getcwd())[0], 'EndoMapper', 'Simulated Sequences')
    
    for subdir in os.listdir(endomapper_dir)[:-1]:
        rgb_image_path = os.path.join(endomapper_dir, subdir, 'inpainted_rgb')
        output_dir = os.path.join(endomapper_dir, subdir, 'rescaled_specularity')
    
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        list_image_framenames = os.listdir(rgb_image_path)

        for image_name in list_image_framenames:
            #print(image_name)
            image_path = os.path.join(rgb_image_path, image_name)
            image = cv2.imread(image_path)
            mask = mask_extraction(image)
            rescaled_spec = scaling_specularities(image, mask)
            cv2.imwrite(os.path.join(output_dir, image_name), rescaled_spec)

    return print('Specularity rescaling complete!')

if __name__ == '__main__':
    main()