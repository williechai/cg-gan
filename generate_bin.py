import numpy as np
from PIL import Image
from tqdm import tqdm
import os

data_path = '/home/williechai/Multi-PIE/'
session_path = ['session01/','session02/','session03/','session04/']

def person2str(person):
    n = person
    result = ''
    result += chr((n / 100) + 48)
    n = n % 100
    result += chr((n / 10) + 48)
    n = n % 10
    result += chr(n + 48)
    return result

def lum2str(lum):
    n = lum
    result = ''
    result += chr((n / 10) + 48)
    n = n % 10
    result += chr(n + 48)
    return result

image_paths = []

for identity in range(1,347):
    for session_idx, session in enumerate(session_path):
        label_path = data_path + session + '07/cropimg_6060/' + person2str(identity) +'_0' + str(session_idx+1) + '_01_051_07.bmp'
        if os.path.exists(label_path):
            for lum in range(0,20):
                for pos in ['_200_', '_190_', '_041_', '_050_', '_051_', '_140_', '_130_', '_080_', '_090_']:
                #for pos in ['_190_', '_041_', '_050_', '_051_', '_140_', '_130_', '_080_']:
                    im_path = data_path + session + lum2str(lum) + '/cropimg_6060/' + person2str(identity) + '_0'+ str(session_idx+1) + '_01' + pos + lum2str(lum) + '.bmp'
                    if not os.path.exists(im_path):
                        print 'error'
                    image_paths.append(im_path)
            break

dataset_path = 'dataset/'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

print 'Generating binary file of iamges, total: ', len(image_paths)
image_array_list = []
for image_path in tqdm(image_paths):
    im = Image.open(image_path)
    image_array_list.append(np.array(im.resize((64, 64)), dtype=np.uint8))

print 'Converting ...'
image_array = np.array(image_array_list, dtype=np.uint8)
print 'To file ...'
image_array.tofile(dataset_path+'data.bin')
print 'Done.'


