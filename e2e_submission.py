
import os
from tqdm import tqdm
import json
from argparse import ArgumentParser
from ocr_recognizer.pipeline import RecognizerPipeline
from pathlib import Path

from utils import *

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--submission_path', type=str, default='/data/ocr_gangnam/private/yoonsoo/hiertext/ensemble_final/submission/validation.jsonl')
    parser.add_argument('--recognizer_path', type=str, default='/data/ocr_gangnam/private/dahyun/hiertext/outputs/recognizer/train/2023-03-20/23-56-36/models/model_latest.pth')
    parser.add_argument('--gt_path', type=str, default='/data/project/bibek/icdar/hier/data/HierText/gt/validation.jsonl')
    parser.add_argument('--gpu_id', type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # with open(args.submission_path, 'r') as f:
    #     submission = json.load(f)
    
    # recognizer = RecognizerPipeline(args.recognizer_path)
    # for ann in tqdm(submission['annotations']):
    #     parsed = parse_annotation(ann,  image_dir='/data/project/bibek/icdar/hier/data/HierText/validation')
    #     texts = recognizer.inference([cv2.cvtColor(parsed['image'], cv2.COLOR_BGR2RGB)], [{'coord': [polygon2rect(polygon) for polygon in parsed['word_polygons']]}])[0]
    #     ctr = 0
    #     for para in ann['paragraphs']:
    #         for line in para['lines']:
    #             for word in line['words']:
    #                 word['text'] = texts[ctr]
    #                 ctr += 1
    
    # with open(args.submission_path, 'w') as f:
    #     json.dump(submission, f)

    if args.gt_path:
        #score_path = str(Path(args.submission_path).parent / 'scores.txt')
        score_path = 'scores.txt'
        os.system(f'python eval.py --gt={args.gt_path} --result={args.submission_path} --output={score_path} --mask_stride=1 --eval_lines --eval_paragraphs --e2e --num_workers=4')
        os.system(f'cat {score_path}-00000-of-00001')