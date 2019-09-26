from crop_cells import crop_cells
import glob, os
import os, glob


def main():
    with open("folders.txt", "r") as f:
        for line in f:
            line = line[2:-1]
            image_dir = "../../data/raw/"
            image_dir = os.path.join(image_dir, line) + "/"
            print(image_dir)

            original_cwd = os.getcwd()

            os.chdir(image_dir)
            os.listdir()

            sample_ids = list(set(["_".join(file.split("_")[:2]) for file in glob.glob("*.png")]))
            os.chdir(original_cwd)
            save_dir = os.path.join("../../data/cropped/", line) + "/"
            min_nuc_size = 100

            cropsize = 64 # this is the RADIUS

            threshold = 0.5 # remove cells of sizes above this quantile (they are likely to be doublets or multiplets)
            for sample_id in sample_ids:
                crop_cells(sample_id, image_dir, save_dir, min_nuc_size, cropsize, threshold)

if __name__ == "__main__":
    main()
