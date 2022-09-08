import os
import cv2
import numpy as np
import data_splitting
import pickle

def main():
    #Creating paths for the RGB and GT Directories to access.
    ucl_rgb_input_dir = os.path.join(os.getcwd(), 'UCL Data Preprocessed', 'RGB') 
    ucl_gt_input_dir = os.path.join(os.getcwd(), 'UCL Data Preprocessed', 'GT')
    ucl_split_path = os.path.join(os.getcwd(), 'UCL Selected Data', 'edges')


    endomapper_rgb_input_dir = os.path.join(os.getcwd(), 'EndoMapper Preprocessed v2', 'RGB')
    endomapper_gt_input_dir = os.path.join(os.getcwd(), 'EndoMapper Preprocessed v2', 'GT')
    endomapper_split_path = os.path.join(os.getcwd(), 'EndoMapper Selected Data', 'edges')

    endomapper_inpaint_only_rgb_input_dir = os.path.join(os.getcwd(), 'EndoMapper Preprocessed v2 Inpainted', 'RGB')
    endomapper_inpaint_only_gt_input_dir = os.path.join(os.getcwd(), 'EndoMapper Preprocessed v2 Inpainted', 'GT')
    endomapper_inpaint_only_split_path = os.path.join(os.getcwd(), 'EndoMapper Selected Data Inpainted Only', 'edges')

    endomapper_spec_only_rgb_input_dir = os.path.join(os.getcwd(), 'EndoMapper Preprocessed v2 Rescaled Specularity', 'RGB')
    endomapper_spec_only_gt_input_dir = os.path.join(os.getcwd(), 'EndoMapper Preprocessed v2 Rescaled Specularity', 'GT')
    endomapper_spec_only_split_path = os.path.join(os.getcwd(), 'EndoMapper Selected Data Rescaled Specularity Only', 'edges')
    #data_splitting_class = data_splitting.DATASET_SPLITTING()

    ucl_data_class_instantiation = data_splitting.DATASET_SPLITTING(ucl_rgb_input_dir, ucl_gt_input_dir, ucl_split_path, True, 290, 145, 145)
    endomapper_data_class_instantiation = data_splitting.DATASET_SPLITTING(endomapper_rgb_input_dir, endomapper_gt_input_dir, endomapper_split_path, True, 300, 150, 150 )
    endomapper_inpaintonly_class_instantiation = data_splitting.DATASET_SPLITTING(endomapper_inpaint_only_rgb_input_dir, endomapper_inpaint_only_gt_input_dir, endomapper_inpaint_only_split_path, True, 300, 150, 150)
    endomapper_speconly_class_instantiation = data_splitting.DATASET_SPLITTING(endomapper_spec_only_rgb_input_dir, endomapper_spec_only_gt_input_dir, endomapper_spec_only_split_path, True, 300, 150, 150)

    ucl_training_sampled_frames, ucl_validation_sampled_frames, ucl_testing_sampled_frames = ucl_data_class_instantiation.frame_sampling() 
    endomapper_training_sampled_frames, endomapper_validation_sampled_frames, endomapper_testing_sampled_frames = endomapper_data_class_instantiation.frame_sampling()
    endomapper_inpaint_training_sampled_frames, endomapper_inpaint_validation_sampled_frames, endomapper_inpaint_testing_sampled_frames = endomapper_inpaintonly_class_instantiation.frame_sampling()
    endomapper_speconly_training_sampled_frames, endomapper_speconly_validation_sampled_frames, endomapper_speconly_testing_sampled_frames = endomapper_speconly_class_instantiation.frame_sampling()


    with open('UCL Training Sampled Frames', 'wb') as fp:
        pickle.dump(ucl_training_sampled_frames, fp)
        fp.close()
    with open('UCL Validation Sampled Frames', 'wb') as fp:
        pickle.dump(ucl_validation_sampled_frames, fp)
    with open('UCL Testing Sampled Frames', 'wb') as fp:
        pickle.dump(ucl_testing_sampled_frames, fp)
    with open('EndoMapper Training Sampled Frames', 'wb') as fp:
        pickle.dump(endomapper_training_sampled_frames, fp)
    with open('EndoMapper Validation Sampled Frames', 'wb') as fp:
        pickle.dump(endomapper_validation_sampled_frames, fp)
    with open('EndoMapper Testing Sampled Frames', 'wb') as fp:
        pickle.dump(endomapper_testing_sampled_frames, fp)

    with open('EndoMapper Inpaint only Training Sampled Frames', 'wb') as fp:
        pickle.dump(endomapper_inpaint_training_sampled_frames, fp)
    with open('EndoMapper Inpaint only Validation Sampled Frames', 'wb') as fp:
        pickle.dump(endomapper_inpaint_validation_sampled_frames, fp)
    with open('EndoMapper Inpaint only Testing Sampled Frames', 'wb') as fp:
        pickle.dump(endomapper_inpaint_testing_sampled_frames, fp)
    
    with open('EndoMapper Spec only Training Sampled Frames', 'wb') as fp:
        pickle.dump(endomapper_speconly_training_sampled_frames, fp)
    with open('EndoMapper Spec only Validation Sampled Frames', 'wb') as fp:
        pickle.dump(endomapper_speconly_validation_sampled_frames, fp)
    with open('EndoMapper Spec only Testing Sampled Frames', 'wb') as fp:
        pickle.dump(endomapper_speconly_testing_sampled_frames, fp)
    

    return
if __name__=='__main__':
    main()