import cv2
import os
import numpy as np
import natsort
import argparse

def parser_arguments():
    parser = argparse.ArgumentParser(description='Args for conversion')
    #parser.add_argument('--simulated_image_dir',
    #                    type=str,
    #                    default = 'Simulated Sequences',
    #                    help='Directory name for the dataset, i.e. HyperKvasir, EndoMapper')
    #parser.add_argument('--sequence_no',
    #                    type=str,
    #                    default = 'seq_0',
    #                    help='Simulated Sequence number')
    #parser.add_argument('--depth_input_dir',
    #                    type=str,
    #                    default='depth_png',
    #                    help='Directory containing the simulated sequence depth map frames')
    #parser.add_argument('--groundtruth_output_dir',
    #                    type=str,
    #                    default='gt')
    #parser.add_argument('--equalised_output_dir',
    #                    type=str,
    #                    default = 'Equalised Grayscale',
    #                    help='Directory containing the equalised grayscale images')
    parser.add_argument('--lower_canny_threshold',
                        type = float,
                        default = 50, #10, #40
                        help='Lower threshold for the OpenCV Canny Edge Detector function')
    parser.add_argument('--upper_canny_threshold',
                        type = float,
                        default = 70, #30, #45 
                        help='Upper threshold for the OpenCV Canny Edge Detector function')
    args = parser.parse_args()
    return args

def image_frame_paths(image_path):
    image_frame_paths = []
    image_frame_names = os.listdir(image_path)
    for frame in image_frame_names:
        image_frame_paths.append(os.path.join(image_path, frame))
    return image_frame_paths

def canny_edge_extraction(image_frame_paths, gt_dir_path, args):#, equalised_img_path):

    for frame in image_frame_paths[:-1]:
        image = cv2.imread(frame, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
      
        alpha = 1.5 #10 # Contrast control (1.0-3.0)
        beta = 15 # Brightness control (0-100)

        image_depth_normalised = (((image[:,:,2])/np.max(image[:,:,2])) * 255).astype(np.uint8) #/np.max(image[:,:,2])) * 255).astype(np.uint8)
        contrast_adjusted = cv2.convertScaleAbs(image_depth_normalised, alpha=alpha, beta=beta)
        
        head_tail = os.path.split(frame)
        edge_map = cv2.Canny(contrast_adjusted, args.lower_canny_threshold, args.upper_canny_threshold)
    
        frame_name = head_tail[1][:-3] + 'png'
        cv2.imwrite(os.path.join(gt_dir_path, frame_name), edge_map)
                
    return print('Finished extracting ground truth edges') 

def main(args):
    endomapper_dir = os.path.join(os.path.split(os.getcwd())[0], 'EndoMapper', 'Simulated Sequences')

    for subdir in os.listdir(endomapper_dir)[:-1]:
        depth_dir_path = os.path.join(endomapper_dir, subdir, 'depth')        
        gt_dir_path = os.path.join(endomapper_dir, subdir, 'gt')
        
        if not os.path.exists(gt_dir_path):
            os.mkdir(gt_dir_path)
        

        image_frame_paths_list = image_frame_paths(depth_dir_path)
        canny_edge_extraction(image_frame_paths_list, gt_dir_path, args) #, rescaled_grayscale_path)

    return print('Converting ground truth to labels complete!')

if __name__=="__main__":
    args = parser_arguments()
    main(args)