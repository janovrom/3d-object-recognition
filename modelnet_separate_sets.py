import os
import sys
import shutil


def move_set(set_file, src_path, dst_path):
    with open(set_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line[0:-1]
            cat_name = line[0:line.rfind("_")]
            in_file = os.path.join(src_path,cat_name,line)+".npy"
            out_dir = os.path.join(dst_path,cat_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, mode=0777)
                
            shutil.copy(in_file, out_dir)
            
    

if __name__ == "__main__":
    #move_set("ModelNet40/modelnet40-normal_numpy/modelnet40_train.txt", "ModelNet40/modelnet40-normal_numpy", "ModelNet40/test")
    if len(sys.argv) < 3:
        print("Wrong use of script. Specify dataset file name, src path, and dst path.")
    else:
        move_set(sys.argv[1], sys.argv[2], sys.argv[3])
