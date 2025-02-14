''' Demo of how to load BERT embeddings for train data.

Will print out embeddings and associated labels
for the SAME 8 randomly chosen examples from load_train_data.py
'''

import numpy as np
import pandas as pd
import os
import textwrap

def load_arr_from_npz(npz_path):
    ''' Load array from npz compressed file given path

    Returns
    -------
    arr : numpy ndarray
    '''
    npz_file_obj = np.load(npz_path)
    arr = npz_file_obj.f.arr_0.copy() # Rely on default name from np.savez
    npz_file_obj.close()
    return arr

if __name__ == '__main__':
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

    # Load BERT embeddings as 2D numpy array
    # Each row corresponds to row of data frames above
    # Each col is one of the H=768 dimensions of BERT embedding    
    xBERT_train_NH = load_arr_from_npz(os.path.join(
        data_dir, 'x_train_BERT_embeddings.npz'))
    assert xBERT_train_NH.ndim == 2

    N, n_cols = x_train_df.shape
    N2, H = xBERT_train_NH.shape
    print("Shape of x_train_df: (%d, %d)" % (N, n_cols))
    print("Shape of y_train_df: %s" % str(y_train_df.shape))
    print("Shape of xBERT_train_NH: %s" % str(xBERT_train_NH.shape))

    # Print out 8 random entries
    tr_text_list = x_train_df['text'].values.tolist()
    prng = np.random.RandomState(101)
    rows = prng.permutation(np.arange(y_train_df.shape[0]))
    for row_id in rows[:8]:
        text = tr_text_list[row_id]
        print("row %5d | %s BY %s | y = %s" % (
            row_id,
            y_train_df['title'].values[row_id],
            y_train_df['author'].values[row_id],
            y_train_df['Coarse Label'].values[row_id],
            ))
        # Pretty print text via textwrap library
        line_list = textwrap.wrap(tr_text_list[row_id],
            width=70,
            initial_indent='  ',
            subsequent_indent='  ')
        print('\n'.join(line_list))
        print("BERT embedding vector (size %d):" % H)
        with np.printoptions(precision=2, edgeitems=4, threshold=50):
            print(xBERT_train_NH[row_id])
        print("")
