'''Script adapted from Soria et al. https://github.com/xavysp/MBIPED'''

from ucl_data_augmentation import augment_data
import os 

def main(dataset_dir):
    augment_both = True  # to augment the RGB and target (edge_map) image at the same time
    augment_data(base_dir=dataset_dir, augment_both=augment_both, use_all_type=True)


if __name__=='__main__':
    base_dir = os.path.join(os.path.split(os.getcwd())[0], 'UCL Selected Data')
    print(base_dir)
    main(dataset_dir=base_dir)