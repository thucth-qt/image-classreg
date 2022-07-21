import cv2 
import matplotlib.pyplot as plt
import numpy as np
import random 
import pandas as pd
import os
from tqdm import tqdm 
from glob import glob


#Inherit from id_exposure

def show_img(amount, part, label, type_id):
    try:
        for side in os.listdir(os.path.join(part, label, type_id)):
            img_names = os.listdir(os.path.join(part, label, type_id, side))
            random.shuffle(img_names)
            for img_name in img_names[:amount]:
                img_path = os.path.join(part, label, type_id, side, img_name)
                print("==========================")
                print(img_path)
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                plt.show()
    except FileNotFoundError as e:
        print(e)
        
# def load_img(part, label, type_id, amount=None):
def load_background(img_paths, amount=None):
    ''' Input: path to label folder'''
    results=[]
    random.shuffle(img_paths)
    for img_path in img_paths[:amount]:
        try:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            results.append([img, os.path.basename(img_path)])
        except Exception as e:
            print(e)
            print(img_path)
    return results

def load_foreground(path):
    results=[]
    for img_path in os.listdir(path):
        try:
            full_path = os.path.join(path,img_path)
            glare = cv2.imread(full_path)[:,:,::-1]
            results.append(glare)
        except Exception as e:
            print(e)
            print(full_path)
    return results

def load_bbox(path):   
    #top, left, width, height
    path2bbox={}
    df = pd.read_csv(path)
    for idx in range(len(df)):
        key = df.iloc[idx][1]
        key = key[key.find("part"):]
        value = df.iloc[idx][0] #string
        value = value[1:-1] #remove "(", ")"
        value = list(map(int, value.split(sep=",")))
        path2bbox[key]=value
    return path2bbox

def resize_scale(src, frac=1.):
    '''
    input: src(H,W,C)
    '''
    frac = float(frac)
    print(src.shape)
    print(frac)
    # resize image
    # if frac < 1:
    #     src = apply_exponential(src, random.randint(2, 6))
    src_resize = cv2.resize(src, dsize=(int(src.shape[1]*frac), int(src.shape[0]*frac)), fx=frac, fy=frac, interpolation=cv2.INTER_CUBIC)
    print(src_resize.shape)
    #center image
    # original_center = (src.shape[0]//2, src.shape[1]//2)
    # resized_center = (src_resize.shape[0]//2, src_resize.shape[1]//2)
    # if frac < 1:
    #     src_resize = np.pad(src_resize, ((0, src.shape[0]-src_resize.shape[0]), (0, src.shape[1]-src_resize.shape[1]), (0, 0)), "constant", constant_values=[(0, 0), (0, 0), (0, 0)])
    # trans_mat = np.array([[1, 0, original_center[1] - resized_center[1]],
    #                      [0, 1, original_center[0] - resized_center[0]]], dtype=np.float64)
    # src_translate = cv2.warpAffine(src_resize, trans_mat, src_resize.shape[1::-1], flags=cv2.INTER_LINEAR)

    # result = src_translate[:src.shape[0], :src.shape[1], :]
    # return result
    return src_resize


def random_translate(src, background_size, bbox):

    ################################
    #   translate in bounding box  #
    ################################
    if bbox:
        t = bbox[0]
        l = bbox[1]
        w = bbox[2]
        h = bbox[3]
    else:
        l = 0
        t = 0
        w = background_size[1]
        h = background_size[0]

    dx = np.random.randint(-w//2, w//2)
    dy = np.random.randint(-h//2, h//2)

    src = np.pad(src, pad_width=[(-min(0, dy), max(0, dy)), (-min(0, dx), max(0, dx)), (0, 0)], mode="constant")
    trans_mat1 = np.array([[1, 0, dx],
                           [0, 1, dy]], dtype=np.float64)
    src = cv2.warpAffine(src, trans_mat1, src.shape[1::-1], flags=cv2.INTER_LINEAR)
    #################################################################
    #  translate center and crop from the center to ouside          #
    #################################################################
    background = np.zeros((max(background_size[0], src.shape[0]), max(background_size[1], src.shape[1]), src.shape[2]))
    background[:src.shape[0], :src.shape[1], :src.shape[2]] = src

    center_id_on_background = (l+w//2, t+h//2)  # x,y
    center_glare = (src.shape[1]//2, src.shape[0]//2)
    trans_mat2 = np.array([[1, 0, center_id_on_background[0] - center_glare[0]],
                           [0, 1, center_id_on_background[1]-center_glare[1]]], dtype=np.float64)
    result = cv2.warpAffine(background, trans_mat2, background.shape[1::-1], flags=cv2.INTER_LINEAR)
    result = result[:background_size[0], :background_size[1], :]
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def change_brightness(src, brightness=100):
    src += 255-src.max()
    brightness /= 100
    src = np.int16(src)
    src = np.round(src*brightness)
    src = np.clip(src, 0, 255)
    return np.uint8(src)


def rotate_image(src, angle):
    size_new = int(np.sqrt(src.shape[0]**2+src.shape[1]**2))
    pad_h = (size_new-src.shape[0])//2
    pad_w = (size_new-src.shape[1])//2
    src = np.pad(src, [(pad_h, pad_h), (pad_w, pad_w), (0, 0)], mode="constant")
    image_center = tuple(np.array(src.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(src, rot_mat, src.shape[1::-1], flags=cv2.INTER_NEAREST)
    return result


def apply_gaussian(src, radius=10):
    src = np.int16(src)
    h, w = src.shape
    y, x = np.mgrid[-h//2+1:h//2+1, -w//2+1:w//2+1]
    kernel = np.exp(-((x**2+y**2)/(2.0*radius**2)))
    src = src*kernel
    src = np.clip(src, 0, 255)
    return np.uint8(src)


def apply_exponential(src, level):
    src = np.int16(src)
    h, w, c = src.shape
    area = min(h//2, w//2)
    y, x = np.mgrid[-h//2+1:h//2+1, -w//2+1:w//2+1]
    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)

    x = x / (w//2)
    kernel = 1 - np.power(np.abs(x), level)
    src = src*kernel
    y = y / (h//2)
    kernel = 1 - np.power(np.abs(y), level)
    src = src*kernel
    src = np.clip(src, 0, 255)
    return np.uint8(src)


def random_color(src, is_color=0.2):
    src = np.float64(src)
    if np.random.choice([False, True], size=1, p=[1-is_color, is_color]):
        # src+= 110  # add color channel for YCrBr
        src[:, :, 0] *= np.random.randint(low=75, high=125)/100
        src[:, :, 1] *= np.random.randint(low=75, high=125)/100
        src[:, :, 2] *= np.random.randint(low=75, high=125)/100
    src = np.clip(src, 0, 255)
    src = np.uint8(src)
    return src


def get_area(bbox):
    "need to reimplement"
    # return bbox['height']*bbox['width']
    return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])

def process(img_and_key, glare, fg_size, path2bbox, percent=1., is_debug=False):
    ##########################
    # normalize glare        #
    ##########################
    img = img_and_key[0]
    path_img = img_and_key[1]
    try:
        if path_img in path2bbox:
            bbox = path2bbox[path_img]
        else:
            bbox = None
    except Exception as e:
        print("img.shape: ", img.shape)
        print("path_img.shape: ", img_and_key[1].shape)
        raise e
    # This for easy purpose, to implement: use exact bbounding box
    bbox = bbox['full']
    if is_debug:
        plt.figure(figsize=(20, 20))
    if is_debug:
        plt.subplot(191).title.set_text("original")
        plt.subplot(191).imshow(img)

    img = img.astype(np.int16)

    ##########################
    # process glare          #
    ##########################
    glare_rgba = cv2.cvtColor(glare, cv2.COLOR_RGB2RGBA)
    glare_gray = cv2.cvtColor(glare, cv2.COLOR_RGB2GRAY)
    glare_rgba[:,:,3][glare_gray<10] =0
    glare_rgba[:,:,3][glare_gray>250] =0

    s_fg = glare.shape[0]*glare.shape[1]
    s_bg = get_area(bbox)
    ratio = percent/s_fg*s_bg
    glare_changed = resize_scale(glare_rgba, frac=np.sqrt(ratio))
    if is_debug:
        plt.subplot(192).title.set_text("resize_scale")
        plt.subplot(192).imshow(glare_changed)
    glare_changed = random_color(glare_changed, is_color=0.1)  # return glare with 3 channels (YCrCb)
    print("max alpha resize_scale: ", glare_changed[:, :, 3].max())

    if is_debug:
        plt.subplot(193).title.set_text("random_color")
        plt.subplot(193).imshow(glare_changed)
    print("max alpha random_color: ", glare_changed[:, :, 3].max())
    # glare_changed = apply_exponential(glare_changed, random.randint(1, 2))
    if is_debug:
        plt.subplot(194).title.set_text("apply_exponential")
        plt.subplot(194).imshow(glare_changed)
    glare_changed = rotate_image(glare_changed, random.randint(0, 360))
    print("max alpha apply_exponential: ", glare_changed[:, :, 3].max())
    if is_debug:
        plt.subplot(195).title.set_text("rotate_image")
        plt.subplot(195).imshow(glare_changed)
    # glare_changed = apply_exponential(glare_changed, random.randint(1, 2))
    if is_debug:
        plt.subplot(196).title.set_text("apply_exponential")
        plt.subplot(196).imshow(glare_changed)
    glare_changed = random_translate(glare_changed, img.shape, bbox)  # return glare with the same size (H,W) of img
    print("max alpha rotate_image: ", glare_changed[:, :, 3].max())
    if is_debug:
        plt.subplot(197).title.set_text("random_translate")
        plt.subplot(197).imshow(glare_changed)
    # glare_changed = apply_exponential(glare_changed, random.randint(1, 2))
    if is_debug:
        plt.subplot(198).title.set_text("apply_exponential")
        plt.subplot(198).imshow(glare_changed)
#     glare_changed[:,:,0] = cv2.equalizeHist(glare_changed[:,:,0])
#     if is_debug:
#         plt.subplot(199).title.set_text("equalizeHist")
#         plt.subplot(199).imshow(glare_changed)
#         plt.show()
    # glare_changed = change_brightness(glare_changed, random.randint(30, 50))
#     glare_changed = glare_changed * np.random.choice(a=[-1,1], size=1,  p = [0.2, 0.8]) #bright or dark
    # import pdb; pdb.set_trace()
    print("max alpha rotate_image: ", glare_changed[:, :, 3].max())
    

    # glare_changed = cv2.cvtColor(glare_changed, cv2.COLOR_RGBA2RGB)
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    ##########################
    # blend glare            #
    ##########################
    slice_0 = slice(0, min(img.shape[0], glare_changed.shape[0]))
    slice_1 = slice(0, min(img.shape[1], glare_changed.shape[1]))

    # glare_changed = glare_changed.astype(np.int16)
    # if img.mean() < 120:
    #     glare_changed[:, :, 0] += 40
    #     glare_changed = apply_exponential(glare_changed, random.randint(2, 7))
    #
    patch_fg = glare_changed[slice_0, slice_1, :]
    patch_bg = img[slice_0, slice_1, :] 

    patch_output = np.ones_like(patch_bg)*255
    # import pdb; pdb.set_trace()
    # import pdb
    # pdb.set_trace()
    # patch_fg[:, :, 3] = cv2.equalizeHist(patch_fg[:, :, 3])
    patch_fg = patch_fg.astype(np.float32)
    print("max alpha: ", patch_fg[:,:,3].max())
    patch_bg = patch_bg.astype(np.float32)
    patch_output[:,:,:3] = patch_fg[:,:,:3]*(patch_fg[:,:,3:]/255) + patch_bg[:,:,:3]*((255-patch_fg[:,:,3:])/255) 

    # patch_bg[patch_fg != 0] = patch_fg[patch_fg != 0]
    
    # import pdb;pdb.set_trace()
    img[slice_0, slice_1, :] =patch_output
    img = np.clip(img, 0, 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # if is_debug:
    #     plt.subplot(192).title.set_text("result")
    #     plt.subplot(192).imshow(img)

    #best
    # img1 = img.copy()
    # patch_fg = glare_changed[slice_0, slice_1, :]
    # patch_bg = img1[slice_0, slice_1, :]
    # patch_bg[patch_fg != 0] = patch_fg[patch_fg != 0]
    # # blended = cv2.addWeighted(src1=patch_fg, alpha=1-0.9, src2=patch_fg, beta=0.9, gamma=0)
    # img1[slice_0, slice_1, :] = patch_bg
    
    # glare_changed = cv2.addWeighted(src1=np.zeros_like(glare_changed), alpha=0.1, src2=glare_changed, beta=0.9, gamma=0)
    # patch_fg = glare_changed[slice_0, slice_1, :]
    # patch_bg = img[slice_0, slice_1, :] 
    # patch_bg[patch_fg != 0] = patch_fg[patch_fg != 0]
    # img[slice_0, slice_1, :] =patch_bg
    # img = cv2.addWeighted(src1=img1, alpha=0.95, src2=img, beta=0.05, gamma=0)
    
    img = np.clip(img, 0, 255).astype(np.uint8)
#     img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

    return [img, img_and_key[1]]


def process_wrapper(img_and_key, glares, fg_size=(112,112), path2bbox=None, is_debug=False):
    times = np.random.choice(a=[1, 2, 3], size=1, p=[0.9, 0.05, 0.05])
    glares_chosen = random.choices(glares, k=int(times))

    img, key = img_and_key[0], img_and_key[1]
    if key not in path2bbox:
        return None
    for glare_chosen in glares_chosen:
        img_and_key = process(img_and_key, glare_chosen, fg_size, path2bbox, is_debug)
    return img_and_key[0]
