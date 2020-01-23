import os
from os.path import join
import numpy as np
import cv2
from side_extractor9 import process_piece as process_piece_9
from side_extractor50 import process_piece as process_piece_50
from functools import partial

import imutils
import argparse


def remover_background(path, filename,x, y , h, w):
    puzzel_stuk = cv2.imread(join(path, filename))

    cv2.imwrite(filename, puzzel_stuk)
    puzzel_stuk = puzzel_stuk[y:y + h, x:x + w]
    postprocess = partial(cv2.blur, ksize=(3, 3))
    if grote_puzzel ==9:
        out_dict = process_piece_9(puzzel_stuk, after_segmentation_func=postprocess, scale_factor=0.2,
                             harris_block_sprize=5, harris_ksize=5, bin_threshold=150,
                             corner_score_threshold=0.5, corner_minmax_threshold=100)
    else:
        out_dict = process_piece_50(puzzel_stuk, after_segmentation_func=postprocess, scale_factor=0.2,
                                   harris_block_size=5, harris_ksize=5, bin_threshold=150,
                                   corner_score_threshold=0.5, corner_minmax_threshold=100)
    return out_dict

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            cut_out = image[y:y + windowSize[1], x:x + windowSize[0]]
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def calculate_hist(img, channel, histSize, ranges):
    hist = cv2.calcHist([img], channel, None, histSize,
                        ranges)
    #[0, 1, 2], None, [32, 32, 32], [1, 256, 1, 256, 1, 256]
    return hist

def vergelijken(puzzelstuk, box):
    if kleur:
        (winW, winH, _) = puzzelstuk.shape
    else:
        (winW, winH) = puzzelstuk.shape
    if stukken_vergelijken == "H":
        if kleur:
            channel, histSize, ranges = [0, 1, 2], [32, 32, 32], [1, 256, 1, 256, 1, 256]
        else:
            channel, histSize, ranges = [0], [256], [1, 256]

        img_hist = calculate_hist(puzzelstuk, channel, histSize, ranges)
        dict_difrance = {}
        for (x, y, cut_out) in sliding_window(box, stepSize=32, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if cut_out.shape[0] != winH or cut_out.shape[1] != winW:
                continue
            cut_out_hist = calculate_hist(cut_out, channel, histSize, ranges)
            dict_difrance[(x, y, winW, winH)] = abs(cv2.compareHist(img_hist, cut_out_hist, cv2.HISTCMP_CORREL))
        dict_difrance = {k: v for k, v in sorted(dict_difrance.items(), key=lambda item: item[1], reverse=True)}
        print("data verwerkt")
        print("img wegschrijven")
        print(dict_difrance)
        eerste_items = dict(list(dict_difrance.items())[0:1])
        eerste_keys = list(eerste_items.keys())
        return (eerste_keys[0], dict_difrance[eerste_keys[0]])

    elif stukken_vergelijken == "M":

        result = cv2.matchTemplate(puzzelstuk, box, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        return (maxLoc[0], maxLoc[1],winW, winH), maxVal

    else:
        edged = cv2.Canny(puzzelstuk, 50, 200)
        edged_box = cv2.Canny(box, 50, 200)
        result = cv2.matchTemplate(edged, edged_box, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        return (maxLoc[0], maxLoc[1],winW, winH),maxVal

def puzzelstuk(path, filename, box, stukken_vergelijken):

    out_dict = remover_background(path, filename, x, y, h, w)
    filename = filename.split(".")
    if kleur:
        puzzelstuk = out_dict["cropped_original"]
        mask = out_dict['mask_3D']
    else:
        puzzelstuk = cv2.cvtColor(out_dict["cropped_original"], cv2.COLOR_BGR2GRAY)
        puzzelstuk = cv2.equalizeHist(puzzelstuk)
        mask = out_dict['mask']
        box = cv2.cvtColor(box_original, cv2.COLOR_BGR2GRAY)
        box = cv2.equalizeHist(box)

    print("puzzelstuk ingelezen")
    puzzelstuk = cv2.bitwise_and(puzzelstuk, mask)
    puzzelstuk = cv2.blur(puzzelstuk, (5, 5))

    stukken_vergelijken = stukken_vergelijken.upper()
    dict_difrance = {}
    if inzoemen:
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            resized = imutils.resize(puzzelstuk, width=int(puzzelstuk.shape[1] * scale))
            temp = vergelijken(resized, box)
            dict_difrance[temp[0]] = temp[1]
    else:
        temp = vergelijken(puzzelstuk, box)
        dict_difrance[temp[0]] = temp[1]
    dict_difrance = {k: v for k, v in sorted(dict_difrance.items(), key=lambda item: item[1], reverse=True)}
    eerste_items = dict(list(dict_difrance.items())[0:1])
    eerste_keys = list(eerste_items.keys())
    box_gray = cv2.cvtColor(box_original, cv2.COLOR_BGR2GRAY)
    for eerste in eerste_keys:
        clone = np.dstack([box_gray, box_gray, box_gray, box_gray])
        # cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
        # (maxLoc[0] + gray.shape[0], maxLoc[1] + gray.shape[1]), (0, 0, 255), 2)
        top_left = (eerste[0], eerste[1])
        bottom_right = (eerste[0] + eerste[2], eerste[1] + eerste[3])
        cv2.rectangle(clone, top_left, bottom_right, (0, 0, 255), 3)

        cv2.imwrite('test1_beste_voor_'+".".join(filename[:-1])+'.png', clone)
        cv2.waitKey(0)
    

# parameters
# coding=utf-8
parser = argparse.ArgumentParser(description='Celine haar demo om een AI te laten puzzelen')
parser.add_argument("--k", default='K', type=str, help="Puzzelstukken in kleur(K) of in zwart wit behandeld moet worden(G)")
parser.add_argument("--z", default='N', type=str, help="Wil je dat grote van de stukjes niet verander(N) of dat hij verkleint en vergroot (Y)")
parser.add_argument("--g", default='9', type=int, help="Aantal stuks: 9 of 50")
parser.add_argument('--s', default='A', type=str, help="Wil je alle stuks (A) of 1 stuk (B)")
args = parser.parse_args()
kleur = args.k
inzoemen = args.z
grote_puzzel = args.g
aantal_stuks = args.s
# kleur = input("In kleur(K) of zwart-wit (G) : ")
# inzoemen = input('Wil je dat grote van de stukjes niet verander(N) of dat hij verkleint en vergroot (Y) ')
# grote_puzzel = int(input("Aantal stuks: 9 of 50: "))
# aantal_stuks= input("Wil je alle stuks (A) of 1 stuk (B)")

#stukken_vergelijken= input("Manieren om puzzel stukken te vergelijken: histogram (H), machtTemplate (M), hoeken op puzzel vergelijken (C): ")

if kleur == 'K':
    kleur = True
else:
    kleur = False

if inzoemen.upper() =='Y':
    inzoemen = True
else:
    inzoemen = False


if grote_puzzel ==9:
    path = "./9st"
    filename_box = "/Doos/0.jpg"
    x = 0
    y = 800
    h = 3000
    w = 5000
else:
    path = "./50st"
    filename_box = "/Doos/0.jpg"
    y = 0
    x = 0
    h = 2700
    w = 3000


box_original = cv2.imread(path+filename_box)
box = box_original.copy()
print("doos ingelezen")

if aantal_stuks =="A":
    paths= path + "/Stuks"
    filenames = os.listdir(paths)
    for filename in filenames:
        puzzelstuk(paths,filename,box , stukken_vergelijken)
else:
    filename = "0.jpg"
    puzzelstuk(path+"/Stuks", filename,box , stukken_vergelijken)



