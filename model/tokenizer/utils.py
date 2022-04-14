import os
import pandas as pd

def shift_challenge_processing(args):

    # WikiMatrix (Processed version)
    with open(os.path.join(args.data_path, 'WikiMatrix.en-ru.txt.en'), 'r') as f:
        wiki_matrix_en = [x.replace('\n', '') for x in f.readlines()]

    with open(os.path.join(args.data_path, 'WikiMatrix.en-ru.txt.ru'), 'r') as f:
        wiki_matrix_en = [x.replace('\n', '') for x in f.readlines()]

    # News
    with open('/HDD/dataset/shift_challenge/news.en', 'r') as f:
        news_en = [x.replace('\n', '') for x in f.readlines()]
        
    with open('/HDD/dataset/shift_challenge/news.ru', 'r') as f:
        news_ru = [x.replace('\n', '') for x in f.readlines()]
        
    with open('/HDD/dataset/shift_challenge/news.en.translatedto.ru', 'r') as f:
        news_en_to_ru = [x.replace('\n', '') for x in f.readlines()]
        
    with open('/HDD/dataset/shift_challenge/news.ru.translatedto.en', 'r') as f:
        news_ru_to_en = [x.replace('\n', '') for x in f.readlines()]

    # W