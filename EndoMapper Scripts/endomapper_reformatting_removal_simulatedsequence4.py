import os
import shutil

def main():
    endomapper_data_v1_dir = os.path.join(os.path.split(os.getcwd())[0], 'EndoMapper Preprocessed')
    endomapper_data_v2_dir = os.path.join(os.path.split(os.getcwd())[0], 'EndoMapper Preprocessed v2')
    if not os.path.isdir(endomapper_data_v2_dir):
        os.makedirs(endomapper_data_v2_dir)

    for dir in os.listdir(endomapper_data_v1_dir):
        #RGB AND GT DIRS
        for subdir in os.listdir(os.path.join(endomapper_data_v1_dir, dir)):
            #INPAINTED AND RESCALED SPECULARITY
            subsubdir_list = os.listdir(os.path.join(endomapper_data_v1_dir, dir, subdir))
            #Remove seq_4:
            subsubdir_list.remove('seq_4')
            for subsubdir in subsubdir_list:
                shutil.copytree(os.path.join(endomapper_data_v1_dir, dir, subdir, subsubdir), os.path.join(endomapper_data_v2_dir, dir, subdir, subsubdir))
    return 

if __name__=='__main__':
    main()