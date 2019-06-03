import errno
import hashlib
import os
import torch
from PIL import Image
from tqdm import tqdm
from utils import makedir_exist_ok

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_img_dataset(dir, extensions, classes_to_labels=None):
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        raise ValueError('Wrong data path')
    if(classes_to_labels is None):
        data = {'img':[]}
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    data['img'].append(path)
    else:
        data = {'img':[],'label':[]}
        for target in sorted(classes_to_labels.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        data['img'].append(path)
                        data['label'].append(classes_to_labels[target])
    return data

def merge_classes(label, class_map):
    for i in range(len(label)):
        if (isinstance(label,list)):
            c = str(label[i])  
        else:
            raise ValueError('label data type not supported')
        if(c in class_map):
            label[i] = class_map[c] 
    return label
       
def make_classes_counts(label,class_sizes):
    classes_counts = torch.zeros(class_sizes)
    for s in label:
        classes_counts[s] += 1
    return classes_counts
    
def gen_bar_updater(pbar):
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True

def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
            )
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
                )

def make_branch_classes_to_labels(branch_classes,idx=0,branch_idx=[],depth=1):
    classes_to_labels = {}
    classes_to_branch_labels = {}
    branch_size = []
    branch_index = 0
    for key in branch_classes:
        if(isinstance(branch_classes[key],list)):
            for i in range(len(branch_classes[key])):
                classes_to_labels[branch_classes[key][i]] = idx
                classes_to_branch_labels[branch_classes[key][i]] = branch_idx + [branch_index,i]
                idx = idx + 1
            branch_size.append(len(branch_classes[key]))
        elif(isinstance(branch_classes[key],dict)):
            sub_classes_to_labels, sub_classes_to_branch_labels, sub_branch_size, depth = make_branch_classes_to_labels(branch_classes[key],idx,[branch_index],depth)
            classes_to_labels = {**classes_to_labels, **sub_classes_to_labels}
            classes_to_branch_labels = {**classes_to_branch_labels, **sub_classes_to_branch_labels}
            branch_size.append(sub_branch_size)
            idx = max(sub_classes_to_labels.values())
        else:
            raise ValueError('Not supported type making branch classes to labels')
        branch_index = branch_index + 1
    depth = depth + 1
    return classes_to_labels, classes_to_branch_labels, branch_size, depth  
        