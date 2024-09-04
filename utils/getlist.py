import os
def get_list(ds):
    if ds == 'DRIVE':
        datalist_trimg = os.listdir(fr'D:\2021\wxr\datasets\{ds}\training\images')
        datalist_trlabel = os.listdir(fr'D:\2021\wxr\datasets\{ds}\training\1st_manual')
        datalist_trmask = os.listdir(fr'D:\2021\wxr\datasets\{ds}\training\mask')
        tr_img_list = [os.path.join(fr'D:\2021\wxr\datasets\{ds}\training\images', i) for i in datalist_trimg]
        tr_label_list = [os.path.join(fr'D:\2021\wxr\datasets\{ds}\training\1st_manual', i) for i in
                         datalist_trlabel]
        tr_mask_list = [os.path.join(fr'D:\2021\wxr\datasets\{ds}\training\mask', i) for i in datalist_trmask]

        datalist_tsimg = os.listdir(fr'D:\2021\wxr\datasets\{ds}\test\images')
        datalist_tslabel = os.listdir(fr'D:\2021\wxr\datasets\{ds}\test\1st_manual')
        datalist_tsmask = os.listdir(fr'D:\2021\wxr\datasets\{ds}\test\mask')

        ts_img_list = [os.path.join(fr'D:\2021\wxr\datasets\{ds}\test\images', i) for i in datalist_tsimg]
        ts_label_list = [os.path.join(fr'D:\2021\wxr\datasets\{ds}\test\1st_manual', i) for i in datalist_tslabel]
        ts_mask_list = [os.path.join(fr'D:\2021\wxr\datasets\{ds}\test\mask', i) for i in datalist_tsmask]
    elif ds == 'hrf':
        datalist_trimg = os.listdir(fr'D:\2021\wxr\datasets\{ds}\training\images')
        datalist_trlabel = os.listdir(fr'D:\2021\wxr\datasets\{ds}\training\1st_manual')
        datalist_trmask = os.listdir(fr'D:\2021\wxr\datasets\{ds}\training\mask')

        tr_img_list = [os.path.join(fr'D:\2021\wxr\datasets\{ds}\training\images', i) for i in datalist_trimg]
        tr_label_list = [os.path.join(fr'D:\2021\wxr\datasets\{ds}\training\1st_manual', i) for i in
                         datalist_trlabel]
        tr_mask_list = [os.path.join(fr'D:\2021\wxr\datasets\{ds}\training\mask', i) for i in datalist_trmask]

        datalist_tsimg = os.listdir(fr'D:\2021\wxr\datasets\{ds}\test\images')
        datalist_tslabel = os.listdir(fr'D:\2021\wxr\datasets\{ds}\test\1st_manual')
        datalist_tsmask = os.listdir(fr'D:\2021\wxr\datasets\{ds}\test\mask')

        ts_img_list = [os.path.join(fr'D:\2021\wxr\datasets\{ds}\test\images', i) for i in datalist_tsimg]
        ts_label_list = [os.path.join(fr'D:\2021\wxr\datasets\{ds}\test\1st_manual', i) for i in datalist_tslabel]
        ts_mask_list = [os.path.join(fr'D:\2021\wxr\datasets\{ds}\test\mask', i) for i in datalist_tsmask]
    elif ds == 'CHASEDB1':
        path = fr'D:\2021\wxr\datasets\{ds}\images'
        txt = fr'D:\2021\wxr\datasets\{ds}\CHASEDB1training.txt'
        f = open(txt, encoding='utf-8')
        temp = []
        for line in f:
            temp.append(line.strip())
        tr_img_list = [os.path.join(path, i)for i in temp]
        tr_label_list = [i.replace('.jpg','_1stHO.png').replace('images','label')for i in tr_img_list]
        tr_mask_list = [i.replace('images','Masks') for i in tr_img_list]

        txt = fr'D:\2021\wxr\datasets\{ds}\CHASEDB1testing.txt'
        f = open(txt, encoding='utf-8')
        temp = []
        for line in f:
            temp.append(line.strip())
        ts_img_list = [os.path.join(path, i)for i in temp]
        ts_label_list = [i.replace('.jpg', '_1stHO.png').replace('images', 'label') for i in ts_img_list]
        ts_mask_list = [i.replace('images', 'Masks') for i in ts_img_list]
    elif ds == 'STARE':
        path = fr'D:\2021\wxr\datasets\{ds}\image'
        txt = fr'D:\2021\wxr\datasets\{ds}\StareTrainingFold1.txt'
        # txt = fr'D:\2021\wxr\datasets\{ds}\StareTrainingFold1.txt'
        f = open(txt, encoding='utf-8')
        temp = []
        for line in f:
            temp.append(line.strip())
        tr_img_list = [os.path.join(path, i) for i in temp]
        tr_label_list = [i.replace('.png', '.ah.png').replace('image', 'labels') for i in tr_img_list]
        # tr_label_list = [i.replace('.png', '.vk.png').replace('image', 'labels') for i in tr_img_list]
        tr_mask_list = [i.replace('image', 'Masks').replace('png','jpg') for i in tr_img_list]

        txt = fr'D:\2021\wxr\datasets\{ds}\StareTestingFold1.txt'
        # txt = fr'D:\2021\wxr\datasets\{ds}\StareTestingFold1.txt'
        f = open(txt, encoding='utf-8')
        temp = []
        for line in f:
            temp.append(line.strip())
        ts_img_list = [os.path.join(path, i) for i in temp]
        # ts_label_list = [i.replace('.png', '.ah.png').replace('image', 'labels') for i in ts_img_list]
        ts_label_list = [i.replace('.png', '.ah.png').replace('image', 'labels') for i in ts_img_list]
        ts_mask_list = [i.replace('image', 'Masks').replace('png','jpg') for i in ts_img_list]
    return tr_img_list,tr_label_list,tr_mask_list,ts_img_list,ts_label_list,ts_mask_list
if __name__ == '__main__':
    tr_img_list, tr_label_list, tr_mask_list, ts_img_list, ts_label_list, ts_mask_list = get_list('CHASEDB1')
    # print(tr_img_list, tr_label_list, tr_mask_list, ts_img_list, ts_label_list, ts_mask_list)
    print(tr_img_list)
    print(tr_label_list)
    print(tr_mask_list)
    print(ts_img_list)
    print(ts_label_list)
    print(ts_mask_list)