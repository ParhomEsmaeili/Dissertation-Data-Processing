import os
import shutil

def main():
    endomapper_data_input_dir = os.path.join(os.path.split(os.getcwd())[0], 'EndoMapper Preprocessed v2')
    endomapper_data_inpainted_dir = os.path.join(os.path.split(os.getcwd())[0], 'EndoMapper Preprocessed v2 Inpainted')
    endomapper_data_specularity_dir = os.path.join(os.path.split(os.getcwd())[0], 'EndoMapper Preprocessed v2 Rescaled Specularity')
    if not os.path.isdir(endomapper_data_inpainted_dir):
        os.makedirs(endomapper_data_inpainted_dir)
    if not os.path.isdir(endomapper_data_specularity_dir):
        os.makedirs(endomapper_data_specularity_dir)

    for dir in os.listdir(endomapper_data_input_dir):
        #RGB AND GT DIRS
        for subdir in os.listdir(os.path.join(endomapper_data_input_dir, dir)):
            #INPAINTED AND RESCALED SPECULARITY
            if subdir == 'Inpainted':
                shutil.copytree(os.path.join(endomapper_data_input_dir, dir, subdir), os.path.join(endomapper_data_inpainted_dir, dir, subdir))
            if subdir == 'Rescaled Specularity':
                shutil.copytree(os.path.join(endomapper_data_input_dir, dir, subdir), os.path.join(endomapper_data_specularity_dir, dir, subdir))
    return 

if __name__=='__main__':
    main()