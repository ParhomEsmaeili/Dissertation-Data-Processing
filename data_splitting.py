import os
import cv2
import numpy as np
import random 
import natsort 

class DATASET_SPLITTING():
    '''
    Class for splitting the dataset 
    '''
    def __init__(
                self,
                rgb_dir_path: str,
                gt_dir_path: str,
                output_path: str,
                random_sampling: bool,
                total_required_training_frames: float,
                total_required_validation_frames: float,
                total_required_testing_frames: float
            ):
        self.rgb_dir_path = rgb_dir_path
        self.gt_dir_path = gt_dir_path
#        self.output_path = output_path
        self.random_sampling = random_sampling
        self.total_required_training_frames = total_required_training_frames
        self.total_required_validation_frames = total_required_validation_frames
        self.total_required_testing_frames = total_required_testing_frames

        self.train_dir_rgb = os.path.join(output_path, 'imgs', 'train', 'rgbr','real')
        self.validation_dir_rgb = os.path.join(output_path, 'imgs', 'validation', 'rgbr', 'real')
        self.test_dir_rgb = os.path.join(output_path, 'imgs', 'test', 'rgbr', 'real')

        self.train_dir_gt = os.path.join(output_path, 'edge_maps', 'train', 'rgbr', 'real')
        self.validation_dir_gt = os.path.join(output_path, 'edge_maps', 'validation', 'rgbr', 'real')
        self.test_dir_gt = os.path.join(output_path, 'edge_maps', 'test', 'rgbr', 'real')

        if not os.path.isdir(self.train_dir_rgb):
            os.makedirs(self.train_dir_rgb)
        if not os.path.isdir(self.validation_dir_rgb):
            os.makedirs(self.validation_dir_rgb)
        if not os.path.isdir(self.test_dir_rgb):
            os.makedirs(self.test_dir_rgb)

        if not os.path.isdir(self.train_dir_gt):
            os.makedirs(self.train_dir_gt)
        if not os.path.isdir(self.validation_dir_gt):
            os.makedirs(self.validation_dir_gt)
        if not os.path.isdir(self.test_dir_gt):
            os.makedirs(self.test_dir_gt)

    def frame_sampling(self):
        #Determine how many non-empty subcategories exist in the dataset, to sample an equal amount of frames from each subcategory.
        total_subcategories = 0
        for subdir in os.listdir(self.rgb_dir_path):
            #subdir_subcategory_quantity = 0 #len(os.listdir(os.path.join(self.rgb_dir_path, subdir)))
            for subcategory in os.listdir(os.path.join(self.rgb_dir_path, subdir)):
                subcategory_list_frames = os.listdir(os.path.join(self.rgb_dir_path, subdir, subcategory))
                if len(subcategory_list_frames) > 0:
                    total_subcategories += 1
        print('There are {} total subcategories'.format(total_subcategories))            

        training_frames_per_subcategory = self.total_required_training_frames/total_subcategories
        validation_frames_per_subcategory = self.total_required_validation_frames/total_subcategories
        testing_frames_per_subcategory = self.total_required_testing_frames/total_subcategories
        
        #Relabelling the images when collating requires counting the number of frames to relabel each corresponding image.
        train_frame_val = 0
        validation_frame_val = 0
        test_frame_val = 0

        training_subcategory_sampled_frames_array = []
        validation_subcategory_sampled_frames_array = []
        testing_subcategory_sampled_frames_array = []
        #Storing the frames which are sampled from each subcategory for later reference.

        for subdir in os.listdir(self.rgb_dir_path):
            for subcategory in os.listdir(os.path.join(self.rgb_dir_path, subdir)):
                #Randomly sampling a 'frames_per_subcategory' quantity of frames from each subcategory.
                training_subcategory_sampled_frames = []
                validation_subcategory_sampled_frames = []
                testing_subcategory_sampled_frames = []
                subcategory_size = len(os.listdir(os.path.join(self.rgb_dir_path, subdir, subcategory)))
                
                #Selecting and extracting training frames.
                if subcategory_size > 0:
                    for frame_num in range(int(training_frames_per_subcategory)):
                        train_frame_val += 1 
                        while True:
                            #randomly sample a frame from length of subcategory list
                            sampled_frame = random.randint(0,subcategory_size - 1)
                            if sampled_frame not in training_subcategory_sampled_frames:
                                training_subcategory_sampled_frames.append(sampled_frame)
                                #Reading training image and writing it to the output directory.
                                rgb_image = cv2.imread(os.path.join(os.path.join(self.rgb_dir_path, subdir, subcategory), os.listdir(os.path.join(self.rgb_dir_path, subdir, subcategory))[sampled_frame]))
                                cv2.imwrite(os.path.join(self.train_dir_rgb, 'Frame' + '{}'.format(train_frame_val) + '.jpg'), rgb_image)
                                gt_image = cv2.imread(os.path.join(os.path.join(self.gt_dir_path, subdir, subcategory), os.listdir(os.path.join(self.gt_dir_path, subdir, subcategory))[sampled_frame]), cv2.IMREAD_GRAYSCALE)
                                cv2.imwrite(os.path.join(self.train_dir_gt, 'Frame' + '{}'.format(train_frame_val) + '.png'), gt_image)

                                break 
                training_subcategory_sampled_frames_array.append(training_subcategory_sampled_frames)
                print('finished subdir')

            
                #Extracting validation frames.
                if subcategory_size > 0:
                    for frame_num in range(int(validation_frames_per_subcategory)):
                        validation_frame_val += 1 
                        while True:
                            #randomly sample a frame from length of subcategory list
                            sampled_frame = random.randint(0,subcategory_size - 1)
                            if sampled_frame not in training_subcategory_sampled_frames:
                                if sampled_frame not in validation_subcategory_sampled_frames:
                                    validation_subcategory_sampled_frames.append(sampled_frame)
                                    #Reading training image and writing it to the output directory.
                                    
                                    rgb_image = cv2.imread(os.path.join(os.path.join(self.rgb_dir_path, subdir, subcategory), os.listdir(os.path.join(self.rgb_dir_path, subdir, subcategory))[sampled_frame]))
                                    cv2.imwrite(os.path.join(self.validation_dir_rgb, 'Frame' + '{}'.format(validation_frame_val) + '.jpg'), rgb_image)
                                    gt_image = cv2.imread(os.path.join(os.path.join(self.gt_dir_path, subdir, subcategory), os.listdir(os.path.join(self.gt_dir_path, subdir, subcategory))[sampled_frame]), cv2.IMREAD_GRAYSCALE)
                                    cv2.imwrite(os.path.join(self.validation_dir_gt, 'Frame' + '{}'.format(validation_frame_val) + '.png'), gt_image)
                                                
                                    break 
                validation_subcategory_sampled_frames_array.append(validation_subcategory_sampled_frames)
                print('finished subdir')
                #Extracting testing frames.
                if subcategory_size > 0:
                    for frame_num in range(int(testing_frames_per_subcategory)):
                        test_frame_val += 1 
                        while True:
                            #randomly sample a frame from length of subcategory list
                            sampled_frame = random.randint(0,subcategory_size - 1)
                            if sampled_frame not in training_subcategory_sampled_frames:
                                if sampled_frame not in validation_subcategory_sampled_frames:
                                    if sampled_frame not in testing_subcategory_sampled_frames:
                                        testing_subcategory_sampled_frames.append(sampled_frame)

                                        #Reading training image and writing it to the output directory.
                                        rgb_image = cv2.imread(os.path.join(os.path.join(self.rgb_dir_path, subdir, subcategory), os.listdir(os.path.join(self.rgb_dir_path, subdir, subcategory))[sampled_frame]))
                                        cv2.imwrite(os.path.join(self.test_dir_rgb, 'Frame' + '{}'.format(test_frame_val) + '.jpg'), rgb_image)
                                        gt_image = cv2.imread(os.path.join(os.path.join(self.gt_dir_path, subdir, subcategory), os.listdir(os.path.join(self.gt_dir_path, subdir, subcategory))[sampled_frame]), cv2.IMREAD_GRAYSCALE)
                                        cv2.imwrite(os.path.join(self.test_dir_gt, 'Frame' + '{}'.format(test_frame_val) + '.png'), gt_image)
                                        break 
                testing_subcategory_sampled_frames_array.append(testing_subcategory_sampled_frames)
                print('finished subdir')
        return training_subcategory_sampled_frames_array, validation_subcategory_sampled_frames_array, testing_subcategory_sampled_frames_array


                       

