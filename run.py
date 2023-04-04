import os

config_name = 'cfg13'

for th_word in [0.4, 0.5]:
    for th_line in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for th_para in [0.5, 0.6, 0.7, 0.8]:
            save_path = f'w_{th_word}_{th_line}_{th_para}'
            
            if os.path.exists(f'./output/{config_name}/{save_path}'):
                print(f'{save_path} already exists...')
                continue
            
            #os.system(f'CUDA_VISIBLE_DEVICES=6 python run_combined.py --sub_path1 output/cfg8/submission/validation.pt --sub_path2 output/cfg7/submission/validation.pt --sub_path3 output/cfg10/submission/validation.pt --sub_path4 output/cfg12/submission/validation.pt --config_name {config_name} --save_path {save_path} --th_word {th_word} --th_line {th_line} --th_para {th_para}')
            
            #os.system(f'CUDA_VISIBLE_DEVICES=5 python predict.py --config_name {config_name} --val_resolution 2048 --val_batch_size 4 --ckpt output/{config_name}/model/last.ckpt --tta')
            #os.system(f'python make_submission.py --config_name {config_name} --save_path {save_path} --th_word {th_word} --th_line {th_line} --th_para {th_para}')
            
            os.system(f'CUDA_VISIBLE_DEVICES=6 python run_combined.py --sub_path1 /data/ocr_gangnam/private/yoonsoo/hiertext/ensemble_final/submission/averaged_validation.pt --config_name {config_name} --save_path {save_path} --th_word {th_word} --th_line {th_line} --th_para {th_para}')
            os.system(f'python eval.py --gt=/data/project/bibek/icdar/hier/data/HierText/gt/validation.jsonl --result=./output/{config_name}/{save_path}/validation.jsonl --output=./output/{config_name}/{save_path}/scores.txt --mask_stride=1 --eval_lines --eval_paragraphs --num_workers=4')
