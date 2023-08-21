import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def create_frames_from_videos(videos_folder: str, images_folder: str, fps='2'):

    # Create temp directory when it does not exist
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    else:
        for file in os.scandir(images_folder):
            os.remove(file.path)

    for filename in os.listdir(videos_folder):
        if (filename.endswith(".mp4")): #or .avi, .mpeg, whatever.
            file_path_video = f'{videos_folder}/{filename}'
            file_path_image = f'{images_folder}/{filename.replace(".mp4", "")}'
            os.system(f'ffmpeg -i {file_path_video} -vf fps={fps} {file_path_image}%03d.jpeg >/dev/null 2>&1')
            print(f'video {filename} is converted to images')
        else:
            continue


def plot_annotated_images(folder: str):

    images = {}
    for img_path in glob.glob('temp/test_set/detections/exp/*.jpeg'):
        images[img_path] = mpimg.imread(img_path)

    # Sort images dict by the image name
    img_paths = list(images.keys())
    img_paths.sort()
    images = {i: images[i] for i in img_paths}

    columns = 4
    plt.figure(figsize=(columns * 15, len(images) / columns * 10))

    for i, img_path in enumerate(images):
        image = images[img_path]
        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.title(img_path)
        plt.imshow(image)