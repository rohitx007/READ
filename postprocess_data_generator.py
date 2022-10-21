import os
# print(id2word)

save_dir = "save"
os.makedirs(save_dir, exist_ok=True)
save_dir = os.path.abspath(save_dir)

def load_data(file_path):
    '''
    # Arguments:
        file_path: str
    # Returns:
        data: list of list of str, data[i] means a sentence, data[i][j] means a
            word.
    '''
    data = []
    for line in open(file_path, encoding='utf-8'):
        words = line.strip().split()
        data.append(words)

    return data

def generates_samples(g_data_path, output_file, id2word):
    '''
    Generate sample sentences to output file
    # Arguments:
        T: int, max time steps
        g_data: Generator Data
        num: int, number of sentences
        output_file: str, path
    '''
    g_data = load_data(g_data_path)

    sentences=[]

    for sentence_id in g_data:
        # print(g_data[:3])
        word_sentence = [id2word[int(word)] for word in sentence_id if (int(word) != 0 and int(word) != 2 and int(word) !=1)]
        sentences.append(word_sentence)

    output_str = ''

    for i in sentences:
        output_str += ' '.join(i) + '\n'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_str)

if __name__ == '__main__':
    generates_samples( g_data_path = os.path.join(save_dir,'evaler_file0.619'),output_file =  os.path.join(save_dir,'generated_data_6.19.txt'), id2word = id2word)