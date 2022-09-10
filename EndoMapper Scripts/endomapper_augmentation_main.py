'''Script adapted from Soria et al. https://github.com/xavysp/MBIPED'''
from endomapper_data_augmentation import augment_data
import os 

def main(dataset_dir):
    augment_both = True  # to augment the RGB and target (edge_map) image at the same time
    augment_data(base_dir=dataset_dir, augment_both=augment_both, use_all_type=True)


if __name__=='__main__':
    base_dir = os.path.join(os.path.split(os.getcwd())[0], 'EndoMapper Selected Data')
    print(base_dir)
    main(dataset_dir=base_dir)

    base_dir2 = os.path.join(os.path.split(os.getcwd())[0], 'EndoMapper Selected Data Inpainted Only')
    print(base_dir2)
    main(dataset_dir=base_dir2)
    
    base_dir3 = os.path.join(os.path.split(os.getcwd())[0], 'EndoMapper Selected Data Rescaled Specularity Only')
    print(base_dir3)
    main(dataset_dir=base_dir3)