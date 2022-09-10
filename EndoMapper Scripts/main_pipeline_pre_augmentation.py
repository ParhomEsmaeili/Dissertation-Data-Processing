#import EndoMapper_gt_extraction 
import os
def main():
    os.system('python EndoMapper_gt_extraction.py')
    os.system('python inpainting_rgb_v2.py')
    os.system('python rescaling_specularity_intensity.py')
    os.system('python endomapper_reformatting.py')
    os.system('python endomapper_reformatting_removal_simulatedsequence4.py')
    os.system('python endomapper_split_specularity_inpainted.py')

if __name__=='__main__':
    main()