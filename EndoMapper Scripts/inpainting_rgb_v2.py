import os
import cv2
import numpy as np
import natsort
import argparse

def parser_arguments():
    parser = argparse.ArgumentParser(description='Args for conversion')
    parser.add_argument('--simulated_image_dir',
                        type=str,
                        default = 'Simulated Sequences',
                        help='Directory name for the dataset, i.e. HyperKvasir, EndoMapper')
    parser.add_argument('--sequence_no',
                        type=str,
                        default = 'seq_0',
                        help='Simulated Sequence number')
    parser.add_argument('--input_dir',
                        type=str,
                        default='rgb',
                        help='Directory containing the simulated sequence rgb frames')
    parser.add_argument('--corrected_output_dir',
                        type=str,
                        default = 'inpainted_rgb',
                        help='Directory containing the equalised rgb images')
    args = parser.parse_args()
    return args

def image_frame_paths(image_path):
    image_frame_paths = []
    image_frame_names = os.listdir(image_path)
    for frame in image_frame_names:
        image_frame_paths.append(os.path.join(image_path, frame))
    return image_frame_paths

def rgb_correction(image_frame_paths, corrected_image_dir_path):

    for frame in image_frame_paths:
        
        image_grayscale = cv2.imread(frame,0)
        image_rgb = cv2.imread(frame)
        sobel3d = cv2.Sobel(image_grayscale,cv2.CV_64F,1,0,ksize=3)/255
        high_deriv_mask = np.where(sobel3d > 0.3, 1, 0)
        mask_1 = np.array(255*high_deriv_mask).astype(np.uint8)    
        #Dilating the mask for inpainting
        kernel = np.ones((5,5), np.uint8)
        dilated_deriv_mask = cv2.dilate(mask_1,kernel,iterations = 1)
        #black pixel mask 
        black_pixel_mask = np.where(image_grayscale < 50, 1, 0).astype(np.uint8)
        dilated_black_pixel_mask = cv2.dilate(black_pixel_mask, kernel, iterations=1)
        #overall_mask 
        final_mask = dilated_deriv_mask*dilated_black_pixel_mask #dilated_non_white_pixel_mask
        inpainted_image = cv2.inpaint(image_rgb, final_mask, 3, cv2.INPAINT_TELEA)
        
        head_tail = os.path.split(frame)
        filename = head_tail[1][:-3]
        output_filename = filename + 'jpg'
        cv2.imwrite(os.path.join(corrected_image_dir_path, output_filename), inpainted_image)
    return print('Finished inpainting RGB sequence')
def main(args):
    endomapper_dir = os.path.join(os.path.split(os.getcwd())[0], 'EndoMapper', 'Simulated Sequences')
    
    for subdir in os.listdir(endomapper_dir)[:-1]:
        rgb_dir_path = os.path.join(endomapper_dir, subdir, 'rgb')
        corrected_dir_path = os.path.join(endomapper_dir, subdir, 'inpainted_rgb')
    
        if not os.path.exists(corrected_dir_path):
            os.mkdir(corrected_dir_path)
        image_frame_paths_list = image_frame_paths(rgb_dir_path)[:-1]
        rgb_correction(image_frame_paths_list, corrected_dir_path)

    return print('Finished inpainting all RGB sequences!')

if __name__=="__main__":
    args = parser_arguments()
    main(args)