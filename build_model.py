""" Build model(s) for identifying scanned PDFs """

from pathlib import Path
import datetime

import PyPDF2
import datefinder
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import progressbar

def get_text_from_pdf(pdf_path: Path) -> str:
    """ Get text from PDF

    :param pdf_path: path to pdf file
    :type pdf_path: Path
    :return: text from pdf file
    :rtype: str
    """
    
    # creating a pdf reader object
    try:
        pdfReader = PyPDF2.PdfFileReader(str(pdf_path)) 
    except PyPDF2.utils.PdfReadError:
        print(f'WARNING: Failed to read {pdf_path}')
        return ''

    # printing number of pages in pdf file 
    numpages = pdfReader.numPages

    page = 0
    text = ''
    while page < numpages:
        # creating a page object 
        pageObj = pdfReader.getPage(0) 

        # extracting text from page 
        text += pageObj.extractText()
        page += 1
    return text

TOP = Path('/Volumes/Documents/Records')

print('Getting top directories...')
dir_paths = [item for item in TOP.glob('*') if item.is_dir()]

data = []
for directory in dir_paths:
    print(f'Working {directory}...')
    file_paths = [
        item for item in directory.rglob('*') 
        if item.is_file() and 
        item.name[0:8].isnumeric() and # starts with date
        item.suffix.lower() == '.pdf'
    ]
    
    for file_path in progressbar.progressbar(file_paths):
        try:
            labeled_date = datetime.datetime.strptime(
                file_path.name[0:8],
                '%Y%m%d',
            )
        except ValueError:
            print(f'WARNING: Failed to parse data in {file_path.name}')
            continue
        text = get_text_from_pdf(file_path)
        # Assuming scans are from 2010 to now
        good_dates = [
            item for item in datefinder.find_dates(text) 
            if item.replace(tzinfo=None) > datetime.datetime(2010, 1, 1).replace(tzinfo=None) and 
            item.date() < datetime.date.today()
        ]
        if len(good_dates) > 1:
            print(f'WARNING: Found more than one date in {file_path} {good_dates}')
            unique_good_dates = set(good_dates)
            best_date = unique_good_dates.pop()
            best_date_qty = good_dates.count(best_date)
            for unique_date in unique_good_dates:
                quantity = good_dates.count(unique_date)
                if quantity > best_date_qty:
                    best_date = unique_date
                    best_date_qty = quantity
            selected_date = best_date
        elif len(good_dates) == 0:
            selected_date = None
        else:
            selected_date = good_dates[0]
        data.append({
            'type': directory.name,
            'selected_date': selected_date,
            'labeled_date': labeled_date,
            'text': text,
            'path': file_path
        })

print('Assessing words...')
word_counts = {}
for item in data:
    text = item['text']
    for word in text.split(' '):
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1
# Words that do not occur more than once are not very unique
keys_to_remove = []
for key in word_counts:
    if word_counts[key] == 1:
        keys_to_remove.append(key)
for key in keys_to_remove:
    word_counts.pop(key)

print('Quantifying words...')
data_df = pd.DataFrame(data)
word_list = []
for word in word_counts:
    word_list.append(word)
    try:
        data_df[f'{word}_present'] = data_df['text'].str.contains(word)
    except:
        print(f'Could not do {word}')
labels = data_df['type']
data_df[['type', 'selected_date', 'labeled_date', 'path']].to_csv('file_data.csv')
data_df = data_df.drop(
    ['type', 'selected_date', 'labeled_date', 'path', 'text'], 
    axis='columns'
)
data_df.to_csv('training_data.csv')
print('Building model...')
scaler = StandardScaler().fit(data_df)
scaled_data = scaler.transform(data_df)
svc_model = SVC()
svc_model.fit(scaled_data, np.ravel(labels))
dump(svc_model, 'scan_id_model.joblib') 
dump(scaler, 'scan_scaler.joblib')
dump(word_list, 'scan_word_list.joblib')

