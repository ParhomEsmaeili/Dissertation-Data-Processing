import os
import cv2
import natsort
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Reformatting')
    parser.add_argument('--dataset',
                        type=str,
                        default='UCLSyntheticDataset',
                        help='The dataset which is being processed.')
    args = parser.parse_args()
    return args

def extract_gt(directory_path, gt_output_dir_path, rgb_output_dir_path):

    length_directory = len(os.listdir(directory_path))
    directory_image_list = natsort.natsorted(os.listdir(directory_path))
    #Half the images are the grayscale and half are rgb
    for index in range(int(length_directory/2)):
        img = cv2.imread(os.path.join(directory_path, directory_image_list[index]), cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, [512,512])
        gt_edge_image = cv2.Canny(resized_img,10,30)
        cv2.imwrite(os.path.join(gt_output_dir_path, 'Frame' + '{}'.format(index) + '.png'), gt_edge_image)

    for index in range(int(length_directory/2), length_directory):
        img = cv2.imread(os.path.join(directory_path, directory_image_list[index]))
        resized_img = cv2.resize(img, [512,512])
        cv2.imwrite(os.path.join(rgb_output_dir_path, 'Frame' + '{}'.format(index - int(length_directory/2)) + '.jpg'), resized_img)
    return print('Subdirectory complete!')

def main(args):
    dataset_dir = os.path.join(os.path.split(os.getcwd())[0], args.dataset)
    output_dir = os.path.join(os.path.split(os.getcwd())[0], 'UCL Data Preprocessed')
    dataset_subdir = os.listdir(dataset_dir)

    for subdir in dataset_subdir:
        subsubdir_list = os.listdir(os.path.join(dataset_dir, subdir))
        for subsubdir in subsubdir_list:
            subsubdir_path = os.path.join(dataset_dir, subdir, subsubdir)
            gt_output_dir_path = os.path.join(output_dir, 'GT', subdir, subsubdir)
            rgb_output_dir_path = os.path.join(output_dir, 'RGB', subdir, subsubdir)
            if not os.path.isdir(gt_output_dir_path):
                os.makedirs(gt_output_dir_path)   
            if not os.path.isdir(rgb_output_dir_path):
                os.makedirs(rgb_output_dir_path)
            extract_gt(subsubdir_path, gt_output_dir_path, rgb_output_dir_path)
        
    return print('Finished reformatting')

if __name__=='__main__':
    args = parse_args()
    main(args)