from random import random
import numpy as np
import torch

from torch.utils.data import Dataset

from PIL import Image

# get this file directory
import os

from data.data import video_tensor_to_gif, video_tensor_to_pil_images

this_file_dir = os.path.dirname(os.path.abspath(__file__))


def debug(msg, *args, **kwargs):
    print("\033[93m")
    print("\n==============================")
    print(msg, *args, **kwargs)
    print("==============================\n")
    print("\033[0m")


def debug_save(msg, *args, **kwargs):
    if "filename" in kwargs:
        filename = kwargs["filename"]
        del kwargs["filename"]
    else:
        filename = "debug.txt"
    with open(f"{this_file_dir}/{filename}", "a") as f:
        print("\n==============================", file=f)
        print(msg, *args, **kwargs, file=f)
        print("==============================\n", file=f)


def tensor2d_to_readable_string(tensor):
    # just get the last 2 dimensions
    tensor_shape = tensor.shape
    if len(tensor_shape) > 2:
        for i in range(len(tensor_shape) - 2):
            tensor = tensor[0]
    max_shape = 20
    output = ""
    for i in range(min(max_shape, tensor.shape[0])):
        for j in range(min(max_shape, tensor.shape[1])):
            output += f"{tensor[i, j]:.4f} "
        output += "\n"
    return output


def create_gif_and_img_seq(videos, name, path):
    stacked_frames = videos[0]
    for video in videos[1:]:
        stacked_frames = torch.cat([stacked_frames, video], dim=2)

    video_tensor_to_gif(stacked_frames.cpu(), path / f"{name}.gif")

    video_seq_pil_images = []
    for video in videos:
        video_seq_pil_images.append(
            video_tensor_to_pil_images(video.cpu(), only_first_image=False)
        )

    # combine the images
    combined_height = video_seq_pil_images[0].height * len(video_seq_pil_images)
    combined_image = Image.new("RGB", (video_seq_pil_images[0].width, combined_height))

    for i, image in enumerate(video_seq_pil_images):
        combined_image.paste(image, (0, i * image.height))

    combined_image.save(path / f"{name}.png")


def put_black_rectangle_on_image(image, x=0, y=70, w=128, h=25):
    # put a black rectangle on the image
    image[:, y : y + h, x : x + w] = 0
    return image


def put_black_rectangle_on_video(video, x=0, y=70, w=128, h=25):
    # put a black rectangle on the images
    video[:, :, y : y + h, x : x + w] = 0
    return video


def put_black_rectangle_on_video_batch(videos, x=0, y=70, w=128, h=25):
    # put a black rectangle on the images
    # debug(lt.lovely(videos))
    videos[:, :, :, y : y + h, x : x + w] = 0
    return videos


class FirstFrameDataset(Dataset):
    def __init__(self, dataset_dir, image_size=(64,)):
        self.dataset_dir = dataset_dir
        self.image_size = image_size

    def __len__(self):
        # get the number of images in the dataset directory
        return len(os.listdir(self.dataset_dir))

    def __getitem__(self, index):
        image_path = os.path.join(self.dataset_dir, f"{index}.jpg")
        image = Image.open(image_path)

        # convert image to torch tensor
        image = np.array(image)
        image = torch.from_numpy(image)
        return image


def fix_seed(seed):
    """
    Args :
        seed : fix the seed
    Function which allows to fix all the seed and get reproducible results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
