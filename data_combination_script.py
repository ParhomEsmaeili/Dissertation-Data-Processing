import os
import shutil

def copy_folders(input_dir, output_dir, dataset_name):
    folder_list = os.listdir(input_dir)
    for subfolder in folder_list:
        shutil.copytree(os.path.join(input_dir, subfolder), os.path.join(output_dir, dataset_name + subfolder))

def main():
    ucl_data_rgb_dir = os.path.join(os.getcwd(), 'UCL Selected Data', 'edges', 'imgs', 'train', 'rgbr', 'aug')
    ucl_data_gt_dir = os.path.join(os.getcwd(), 'UCL Selected Data', 'edges', 'edge_maps', 'train', 'rgbr', 'aug')

    endomapper_both_subset_rgb_dir = os.path.join(os.getcwd(), 'EndoMapper Selected Data', 'edges', 'imgs', 'train', 'rgbr', 'aug')
    endomapper_both_subset_gt_dir = os.path.join(os.getcwd(), 'EndoMapper Selected Data', 'edges', 'edge_maps', 'train', 'rgbr', 'aug')
    endomapper_inpaint_rgb_dir = os.path.join(os.getcwd(), 'EndoMapper Selected Data Inpainted Only', 'edges', 'imgs', 'train', 'rgbr', 'aug')
    endomapper_inpaint_gt_dir = os.path.join(os.getcwd(), 'EndoMapper Selected Data Inpainted Only', 'edges', 'edge_maps', 'train', 'rgbr', 'aug')
    endomapper_spec_rgb_dir = os.path.join(os.getcwd(), 'EndoMapper Selected Data Rescaled Specularity Only', 'edges', 'imgs', 'train', 'rgbr', 'aug')
    endomapper_spec_gt_dir = os.path.join(os.getcwd(), 'EndoMapper Selected Data Rescaled Specularity Only', 'edges', 'edge_maps', 'train', 'rgbr', 'aug')

    combined_rgb_output_dir = os.path.join(os.getcwd(), 'Combined Selected Data', 'edges', 'imgs', 'train', 'rgbr', 'aug')
    combined_gt_output_dir = os.path.join(os.getcwd(), 'Combined Selected Data', 'edges', 'edge_maps', 'train', 'rgbr', 'aug')

    combined_with_inpaint_rgb_output_dir = os.path.join(os.getcwd(), 'Combined Selected Data v2', 'edges', 'imgs', 'train', 'rgbr', 'aug')
    combined_with_inpaint_gt_output_dir = os.path.join(os.getcwd(), 'Combined Selected Data v2', 'edges', 'edge_maps', 'train', 'rgbr', 'aug')

    combined_with_respec_rgb_output_dir = os.path.join(os.getcwd(), 'Combined Selected Data v3', 'edges', 'imgs', 'train', 'rgbr', 'aug')
    combined_with_respec_gt_output_dir = os.path.join(os.getcwd(), 'Combined Selected Data v3', 'edges', 'edge_maps', 'train', 'rgbr', 'aug')

    
    if not os.path.isdir(combined_rgb_output_dir):
        os.makedirs(combined_rgb_output_dir)
    if not os.path.isdir(combined_gt_output_dir):
        os.makedirs(combined_gt_output_dir)
    
    if not os.path.isdir(combined_with_inpaint_rgb_output_dir):
        os.makedirs(combined_with_inpaint_rgb_output_dir)
    if not os.path.isdir(combined_with_inpaint_gt_output_dir):
        os.makedirs(combined_with_inpaint_gt_output_dir)

    if not os.path.isdir(combined_with_respec_rgb_output_dir):
        os.makedirs(combined_with_respec_rgb_output_dir)
    if not os.path.isdir(combined_with_respec_gt_output_dir):
        os.makedirs(combined_with_respec_gt_output_dir)
    
    copy_folders(ucl_data_rgb_dir, combined_rgb_output_dir, 'ucl')
    copy_folders(ucl_data_gt_dir, combined_gt_output_dir, 'ucl')
    copy_folders(endomapper_both_subset_rgb_dir, combined_rgb_output_dir, 'endomapper')
    copy_folders(endomapper_both_subset_gt_dir, combined_gt_output_dir, 'endomapper')

    copy_folders(ucl_data_rgb_dir, combined_with_inpaint_rgb_output_dir, 'ucl')
    copy_folders(ucl_data_gt_dir, combined_with_inpaint_gt_output_dir, 'ucl')
    copy_folders(endomapper_inpaint_rgb_dir, combined_with_inpaint_rgb_output_dir, 'endomapper')
    copy_folders(endomapper_inpaint_gt_dir, combined_with_inpaint_gt_output_dir, 'endomapper')

    copy_folders(ucl_data_rgb_dir, combined_with_respec_rgb_output_dir, 'ucl')
    copy_folders(ucl_data_gt_dir, combined_with_respec_gt_output_dir, 'ucl')
    copy_folders(endomapper_spec_rgb_dir, combined_with_respec_rgb_output_dir, 'endomapper')
    copy_folders(endomapper_spec_gt_dir, combined_with_respec_gt_output_dir, 'endomapper')

    return

if __name__=='__main__':
    main()