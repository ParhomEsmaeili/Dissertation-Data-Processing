import os
import cv2
import numpy as np
import natsort

def dataset_reformatting(input_dir, output_dir, gt_bool):
    frame_list = os.listdir(input_dir) 
    sorted_frame_list = natsort.natsorted(frame_list)
    subdir_size = len(sorted_frame_list)

    for index in range(subdir_size):
        if gt_bool:
            img = cv2.imread(os.path.join(input_dir, sorted_frame_list[index]), cv2.IMREAD_GRAYSCALE)
            output_path = os.path.join(output_dir, 'Frame{}.png'.format(index + 1))
        else:
            img = cv2.imread(os.path.join(input_dir, sorted_frame_list[index]))
            output_path = os.path.join(output_dir, 'Frame{}.jpg'.format(index + 1))
        resized_img = cv2.resize(img, [512, 512])
        cv2.imwrite(output_path, resized_img)
    return

def main():
    endomapper_dir = os.path.join(os.path.split(os.getcwd())[0], 'EndoMapper', 'Simulated Sequences')
    output_directory = os.path.join(os.path.split(os.getcwd())[0], 'EndoMapper Preprocessed')
    for subdir in os.listdir(endomapper_dir)[:-1]:
        inpainted_rgb_subdir = os.path.join(endomapper_dir, subdir, 'inpainted_rgb')
        rescaled_specularity_subdir = os.path.join(endomapper_dir, subdir, 'rescaled_specularity')
        gt_subdir = os.path.join(endomapper_dir, subdir, 'gt')

        output_rgb_inpainted_subdir = os.path.join(output_directory, 'RGB', 'Inpainted', subdir)
        output_rgb_rescaled_specularity_subdir = os.path.join(output_directory, 'RGB', 'Rescaled Specularity', subdir)
        output_gt_inpainted_subdir = os.path.join(output_directory, 'GT', 'Inpainted', subdir)
        output_gt_rescaled_specularity_subdir = os.path.join(output_directory, 'GT', 'Rescaled Specularity', subdir)

        if not os.path.isdir(output_rgb_inpainted_subdir):
            os.makedirs(output_rgb_inpainted_subdir)
        if not os.path.isdir(output_rgb_rescaled_specularity_subdir):
            os.makedirs(output_rgb_rescaled_specularity_subdir)
        if not os.path.isdir(output_gt_inpainted_subdir):
            os.makedirs(output_gt_inpainted_subdir)
        if not os.path.isdir(output_gt_rescaled_specularity_subdir):
            os.makedirs(output_gt_rescaled_specularity_subdir)
        
        #Reformatting the datasets into a better processed dataset.
        dataset_reformatting(inpainted_rgb_subdir, output_rgb_inpainted_subdir, False)
        dataset_reformatting(rescaled_specularity_subdir, output_rgb_rescaled_specularity_subdir, False)
        dataset_reformatting(gt_subdir, output_gt_inpainted_subdir, True)
        dataset_reformatting(gt_subdir, output_gt_rescaled_specularity_subdir, True)



    return

if __name__ == '__main__':
    
    main()