"""
Main file for generate Federated Multiview Dataset

################################
There are two steps to generate the sample dataset
1. Generate the images from blender given camera view parameters
2. Postprocess these images for desired object size and image size

Usage:
1. first set BLENDER_PATH variable accordingly
2. specify model_dir (.off files), target_dir and action (all, generate, postprocess) in command line

Notice:
1. "raw" png images  will be generated in target_dir
2. post-processed images will be in the location: "target_dir"+"_postprocessed"
3. in the raw images, object to image ratio may be improper and different w.r.t views, thus postprocessing is
introduced to make them having similar size
4. phong.blend controls the image effect, be careful when modify it
"""


from subprocess import check_output
import os
import glob
import cv2
import numpy as np
import argparse


# !!!!! CHANGE THIS according to your system settings
BLENDER_PATH = "D:/Program Files/blender-2.79b-windows64/blender.exe"



def test_run():
    model_path = ".\\single_off_samples\\airplane_0001.off"
    dst_dir = ".\\single_samples_MV"
    run_one_time(model_path, dst_dir)


def run_one_time(model_path, dst_dir, phi=60, theta_interval=30, phi_offset=0, theta_offset=0):
    task = "\"{}\" phong.blend --background --python phong.py -- {} {} {} {} {} {}".\
        format(BLENDER_PATH, model_path, dst_dir, phi, theta_interval, phi_offset, theta_offset)
    check_output(task, shell=True)


def generate_dataset(src_root_dir, target_root_dir, phi=60, theta_interval=30):
    data_types = ["train", "test"]
    for item in os.listdir(src_root_dir):
        sub_dir = os.path.join(src_root_dir, item)
        if os.path.isdir(sub_dir):
            for dt in data_types:
                src_dir = os.path.join(sub_dir, dt)
                if os.path.exists(src_dir):
                    target_dir = os.path.join(target_root_dir, item, dt)
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)

                    off_files = glob.glob(os.path.join(src_dir, "*.off"))

                    # generate list file
                    with open("temp.txt", "w") as f:
                        for i, off in enumerate(off_files):
                            f.write("{}\n".format(off))
                    # run generated list file
                    run_one_time("temp.txt", target_dir, phi, theta_interval)
                    os.remove("temp.txt")


def process_one_batch(data_dir, target_dir, item_id, phi=60):
    """
    Get a batch of images and do the postprocessing
    """
    # get file names
    img_files = glob.glob(os.path.join(data_dir, "{}_{:03d}_*.png".format(item_id, phi)))

    l_max = None
    for item in img_files:
        img_gray = cv2.imread(item, 0)

        ret, img_gray_thre = cv2.threshold(img_gray, 254, 255, cv2.THRESH_BINARY_INV)

        x, y, w, h = cv2.boundingRect(img_gray_thre)
        if l_max is None:
            l_max = max(w, h)
        else:
            l_max = max(l_max, max(w, h))

    for item in img_files:
        post_process_one_image(item, target_dir, l_max)


def post_process_one_image(img_path, target_dir, l_max, target_size=224):
    """
    Get one generated image, and adapt it to the target scale
    """
    # load the image
    img = cv2.imread(img_path)
    #img_height, img_witdh = img.shape[:2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # thresholding
    ret, img_gray_thre = cv2.threshold(img_gray,254,255,cv2.THRESH_BINARY_INV)

    # get the boundingbox
    x, y, w, h = cv2.boundingRect(img_gray_thre)
    l = max(w, h)

    # compute resize ratio
    resize_ratio = l/l_max

    # compute cropping size
    if w > h:
        dif = (w - h) // 2
        y = y - dif
        l = w
    else:
        dif = (h - w) // 2
        x = x - dif
        l = h

    obj = img[y:y+l,x:x+l,:]
    # in case if obj is empty
    if l != 0:
        new_size = int(204*resize_ratio)
        obj_resized = cv2.resize(obj, (new_size, new_size), interpolation=cv2.INTER_AREA)
        pad_left = (target_size - new_size)//2
        pad_right = target_size - new_size - pad_left
        obj_full = np.pad(obj_resized, ((pad_left, pad_right),(pad_left, pad_right), (0, 0)), 'constant',
                          constant_values=(255, 255))
    else:
        obj_full = np.ones((255,255,3)) * 255
    # write the image to the disk
    cv2.imwrite(os.path.join(target_dir, os.path.basename(img_path)), obj_full,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 1])


def get_class_and_index(entries):
    object_lists = []
    for item in entries:
        temps = os.path.basename(item).split("_")
        if len(temps) == 6:
            obj = "{}_{}_{}".format(temps[0], temps[1], temps[2])
        else:
            obj = "{}_{}".format(temps[0], temps[1])
        object_lists.append(obj)
    object_lists_unique = set(object_lists)
    ordered_list = list(object_lists_unique)
    ordered_list.sort()
    return ordered_list


def post_process_images(src_root_dir, target_root_dir):
    data_types = ["train", "test"]
    for item in os.listdir(src_root_dir):
        sub_dir = os.path.join(src_root_dir, item)
        if os.path.isdir(sub_dir):
            for dt in data_types:
                src_dir = os.path.join(sub_dir, dt)
                if os.path.exists(src_dir):
                    target_dir = os.path.join(target_root_dir, item, dt)
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    img_files = glob.glob(os.path.join(src_dir, "*.png"))

                    # get objects from image files
                    objs = get_class_and_index(img_files)

                    # post process
                    for obj_ind in objs:
                        process_one_batch(src_dir, target_dir, obj_ind, phi=60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--target_dir', required=True)
    parser.add_argument('--action', type=str, default="all", choices=["all", "generate", "postprocess"],
                        help='actions to take')
    parser.add_argument('--phi', type=int, default=60, help='phi')
    parser.add_argument('--theta_interval', type=int, default=30, help='theta interval')

    # process parser arguments
    args = parser.parse_args()
    model_dir = args.model_dir
    target_dir = args.target_dir
    action = args.action
    phi = args.phi
    theta_interval = args.theta_interval

    GENERATE = False
    POSTPROCESS = False
    if action == "all":
        GENERATE = True
        POSTPROCESS = True
        postprocess_dir = target_dir + "_postprocessed"
    if action == "generate":
        GENERATE = True
    if action == "postproces":
        POSTPROCESS = True
        postprocess_dir = target_dir

    # generate images of various views from 3D model
    if GENERATE:
        print("Start generating images from: {}".format(model_dir))
        generate_dataset(model_dir, target_dir, phi, theta_interval)
        print("Images are generated to: {}".format(target_dir))

    # process these images to align objects size
    if POSTPROCESS:
        if not os.path.exists(postprocess_dir):
            os.makedirs(postprocess_dir)

        print("Start postprocessing from: {}".format(target_dir))
        post_process_images(target_dir, postprocess_dir)
        print("Images are postprocessed to: {}".format(postprocess_dir))


if __name__ == "__main__":
    main()
