import json
import os
import sys
import warnings

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import transforms

from efficientnet_pytorch import EfficientNet
from mmdet.apis import inference_detector, init_detector
from collections import Counter

from model import Model
from utils import AttnLabelConverter

from opt import Opt

opt = Opt()
TEST_IMAGES_PATH, SAVE_PATH = sys.argv[1:]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEGM_MODEL_PATH = "segm-model.pth"
OCR_RUS_MODEL_PATH = "ocr_model_rus.pth"
OCR_ENR_MODEL_PATH = "ocr_model_eng.pth"
LANG_DETECT_MODEL_PATH = "lang_detect_model.pth"

BATCH_SIZE = 64
MINI_BATCH = 12


def get_contours_from_mask(mask, min_area=5):
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_list = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            contour_list.append(contour)
    return contour_list


def get_larger_contour(contours):
    larger_area = 0
    larger_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > larger_area:
            larger_contour = contour
            larger_area = area
    return larger_contour


class SEGMpredictor:
    def __init__(self, model_path):
        config_path = "cascade_mask_rcnn_r2_101_fpn_20e_coco.py"
        self.model = init_detector(config_path, model_path, device='cuda:0')

    def __call__(self, imgs):
        results = inference_detector(self.model, imgs)
        batch_contours = []

        for idx in range(len(imgs)):
            prediction = results[idx][1][0]  # img idx, bbox/segm, type_idx
            contours = []
            for pred in prediction:
                contour_list = get_contours_from_mask(pred)
                contours.append(get_larger_contour(contour_list))
            batch_contours.append(contours)

        return batch_contours


def get_converter(lang):
    if lang == 0:
        chars = " !\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_{|}~«»ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№I"
        opt.num_class = 114
    else:
        opt.num_class = 98
        chars = " !\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_{|}~«»ABCDEFGHIJKLMOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz№"

    return len(chars), AttnLabelConverter(chars)
    #  !"#&\'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXY[]_abcdefghijklmnopqrstuvwxyz%№


class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        img = self.data[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (100, 32))
        img_as_tensor = self.to_tensor(img).view(1, 32, 100)
        img_as_tensor = torch.tensor(img_as_tensor, dtype=torch.float32)

        return img_as_tensor

    def __len__(self):
        return len(self.data)


def predict_ocr(images, model, tokenizer, device):
    batch_size = BATCH_SIZE
    data = ImageDataset(images)  # use RawDataset
    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    model.eval()
    predictions = []
    with torch.no_grad():
        for image_tensors in loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device).float()

            length_for_pred = torch.IntTensor([25] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, 25 + 1).fill_(0).to(device)

            preds = model(image, text_for_pred, is_train=False)

            _, preds_index = preds.max(2)
            preds_str = tokenizer.decode(preds_index, length_for_pred)
            for p in preds_str:
                predictions.append(p)
    return predictions


class OcrPredictor:
    def __init__(self, model_path, device="cuda", lang=0):
        self.device = torch.device(device)
        self.tokens, self.tokenizer = get_converter(lang)
        # load model
        self.model = Model(opt)
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(model_path, map_location=device))

    def __call__(self, images):
        if isinstance(images, (list, tuple)):
            one_image = False
        elif isinstance(images, np.ndarray):
            images = [images]
            one_image = True
        else:
            raise Exception(
                f"Input must contain np.ndarray, "
                f"tuple or list, found {type(images)}."
            )

        pred = predict_ocr(images, self.model, self.tokenizer, self.device)

        if one_image:
            return pred[0]
        else:
            return pred


def predict_lang(images, model, device):
    data = ImageDataset(images)  # use RawDataset
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=MINI_BATCH,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    predictions = []
    model.eval()
    with torch.no_grad():
        for image_tensors in loader:
            image_tensors = image_tensors.view((MINI_BATCH, 1, 32, 100)).float()
#             print(image_tensors.shape)
            preds = model(image_tensors.to(device))
            preds = torch.round(torch.sigmoid(preds.view(-1)))
            for p in preds:
                predictions.append(p)

    return predictions


class LanguagePredictor:
    def __init__(self, model_path, device='cuda'):
        self.model = EfficientNet.from_pretrained('efficientnet-b7', in_channels=1, num_classes=1)
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.device = device

    def __call__(self, images):
        if isinstance(images, (list, tuple)):
            one_image = False
        elif isinstance(images, np.ndarray):
            images = [images]
            one_image = True
        else:
            raise Exception(
                f"Input must contain np.ndarray, "
                f"tuple or list, found {type(images)}."
            )

        pred = predict_lang(images, self.model, self.device)
        return pred


def get_image_visualization(img, pred_data, fontpath, font_koef=50):
    h, w = img.shape[:2]
    font = ImageFont.truetype(fontpath, int(h / font_koef))
    empty_img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(empty_img)

    for prediction in pred_data["predictions"]:
        polygon = prediction["polygon"]
        pred_text = prediction["text"]
        cv2.drawContours(img, np.array([polygon]), -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(np.array([polygon]))
        draw.text((x, y), pred_text, fill=0, font=font)

    vis_img = np.array(empty_img)
    vis = np.concatenate((img, vis_img), axis=1)
    return vis


def crop_img_by_polygons(img, polygons):
    # https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
    dsts = []
    for polygon in polygons:
        pts = np.array(polygon)
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        croped = img[y: y + h, x: x + w].copy()
        pts = pts - pts.min(axis=0)
        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        dsts.append(cv2.bitwise_and(croped, croped, mask=mask))
    return dsts


class PipelinePredictor:
    def __init__(self, segm_model_path, ocr_rus_model_path, ocr_eng_model_path, lang_detect_model_path):
        self.segm_predictor = SEGMpredictor(model_path=segm_model_path)
        self.ocr_rus_predictor = OcrPredictor(model_path=ocr_rus_model_path, lang=0)
        self.ocr_eng_predictor = OcrPredictor(model_path=ocr_eng_model_path, lang=1)
        self.lang_predictor = LanguagePredictor(model_path=lang_detect_model_path)
        # print("Initialized")

    def __call__(self, imgs, img_paths):
        # print("Called")
        outputs = []
        all_contours = self.segm_predictor(img_paths)
        # print("Segmented contours")
        for contours, img in zip(all_contours, imgs):
            output = {'predictions': []}
            if contours is not None:
                crops = crop_img_by_polygons(img, contours)
                small_batch = crops[:MINI_BATCH]
                img_lang = Counter(self.lang_predictor(small_batch)).most_common(1)[0][0]
                # print(f"Got image lang ({img_lang})")
                if img_lang == 0:
                    pred_texts = self.ocr_rus_predictor(crops)
                else:
                    pred_texts = self.ocr_eng_predictor(crops)
                
                # print("Got image OCR")

                for (contour, pred_text) in zip(contours, pred_texts):
                    s_idx = pred_text.find("[s]")
                    pred_text = pred_text[:s_idx]
                    output["predictions"].append(
                        {
                            "polygon": [[int(i[0][0]), int(i[0][1])] for i in contour],
                            "text": pred_text,
                        }
                    )
                    
                outputs.append(output)
        # print("Batch pred finished")
        return outputs


class SegmentDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.paths = [path for path in os.listdir(root)]

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.root, self.paths[index]))
        return [image, [self.root, self.paths[index]]]

    def __len__(self):
        return len(self.paths)


def my_collate(batch):
    data = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    return [data, paths]


def main():
    pipeline_predictor = PipelinePredictor(
        segm_model_path=SEGM_MODEL_PATH,
        ocr_rus_model_path=OCR_RUS_MODEL_PATH,
        ocr_eng_model_path=OCR_ENR_MODEL_PATH,
        lang_detect_model_path=LANG_DETECT_MODEL_PATH
    )
    pred_data = {}

    dataset = SegmentDataset(TEST_IMAGES_PATH)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=my_collate,
        pin_memory=True)

    for images, images_paths in dataloader:
        predicts = pipeline_predictor(images, ["/".join(path) for path in images_paths])
        for (i, img_path) in enumerate(images_paths):
            pred_data[img_path[1]] = predicts[i]

    with open(SAVE_PATH, "w") as f:
        json.dump(pred_data, f)


if __name__ == "__main__":
    main()
