#!/usr/bin/env python
# coding: utf-8

# 請先建立一個叫做output的資料夾, 擺在pdf檔案下面一層 #
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PyPDF2
from PyPDF2.filters import FlateDecode
from PIL import Image
from skimage.transform import resize
from skimage.measure import block_reduce
import datetime

def rgb2gray(img):
    """
    Input: RGB img (h, w, c)       -- 輸入圖片檔要有三維(長, 寬, RGB)
    Return: (h, w)                 -- 輸出圖檔只有二維(長, 寬)
    """
    img = img.astype(np.float)
    # img = np.dot(img[..., :], [0.299, 0.587, 0.114])
    img = img[..., 0]                # 因為原始的圖就是灰階的, 所以第三維度(RGB)的地方只取其中一個代表就好(此時是2維陣列, 數值代表顏色)
    np.place(img, img >= 224, 255)   # 大於等於224的數值都丟給255(純白)
    np.place(img, img < 32, 0)       # 小於32的數值都丟給0(純黑)
    return img.astype(np.uint8)      # 其餘的灰色部分保留後回傳灰階的圖(且為整數回傳)


def get_orig_point(img):
    """
    Input: grayscale img (h, w)
    Return: (y, x)                   -- 找出原點(y, x)
    """
    not_white = (img == 0).astype(np.int)      # 找出純黑色的點

    x_sum = np.sum(not_white, axis=1)          # 用累加的方式找到哪個高度(寬度)的位置上有最多的黑色點(也就是x,y軸)
    x_argsort = np.argsort(x_sum)[::-1]
    orig_y = x_argsort[0]
    for i in range(x_argsort.shape[0]):
        if x_argsort[i] - x_argsort[i + 1] == 1:
            if x_sum[x_argsort[i]] - x_sum[x_argsort[i + 1]] <= 10:
                orig_y = min(x_argsort[i], x_argsort[i + 1])
        else:
            break

    y_sum = np.sum(not_white, axis=0)
    y_argsort = np.argsort(y_sum)[::-1]

    orig_x = y_argsort[0]
    for i in range(y_argsort.shape[0]):
        if np.abs(y_argsort[i] - y_argsort[i + 1]) == 1:
            if y_sum[y_argsort[i]] - y_sum[y_argsort[i + 1]] <= 10:
                orig_x = max(y_argsort[i], y_argsort[i + 1])
        else:
            break

    return (orig_y, orig_x)


def get_end_point(img):    # 找x, y軸的長度
    """
    Input:
        img: (h, w)
        orig_p: (y, x)

    Return:
        y_max, x_max
    
    """
    is_white = True                  
    y_max = 0
    while is_white:
        if img[y_max, 0] == 0:
            is_white = False
        else:
            y_max += 1

    is_white = True
    x_max = img.shape[1] - 1
    while is_white:
        if img[img.shape[0] - 1, x_max] == 0:
            is_white = False
        else:
            x_max -= 1
    return [y_max, x_max]


def get_grid_line_ixd(img, end_p):     # 找出虛線
    grid_x_idx = []
    for i in range(img.shape[1]):
        if not np.sum((img[:end_p[0] - 1, i] != 255).astype(np.int)) == 0:
            grid_x_idx.append(i)
    return grid_x_idx


def remove_grid_line(img, grid_x_idx):   # 去除虛線
    for i in grid_x_idx:
        np.place(img[:-1, i], img[:-1, i] <= 127, 255)


def get_mid_values(img, i, v):
    value_idx = np.where(img[:-2, i] == v)[0]

    if value_idx.shape[0] == 0:
        return None
    mid_idx = np.percentile(value_idx, 50).astype(np.int)

    return mid_idx

def fill_with_zero():
    pass


def extract_bar_vales(path='./chart_1.jpg', y_range=(0, 60), x_range=(0, -1)):
    """
    Return: np.array [y axis, v0, v2, ..., vn]
    """

    img = plt.imread(path)
    img = rgb2gray(img)

    orig_p = get_orig_point(img)

    img = img[:orig_p[0] + 1, orig_p[1]:]

    end_p = get_end_point(img)

    grid_x_idx = get_grid_line_ixd(img, end_p)

    remove_grid_line(img, grid_x_idx)

    img = img[end_p[0]:, :end_p[1]]

    with_value_idx = np.where(img[-2, :] != 255)[0]

    bar_values = []
    for i in with_value_idx:
        bar_values.append(img.shape[0] - min(np.where(img[:, i] == 0)[0]) - 1)

    res = np.zeros(img.shape[1])
    res[with_value_idx] = np.array(bar_values)

    y_slope = (y_range[1] - y_range[0]) / img.shape[0]

    res = res * y_slope + y_range[0]

    return res



def extrat_lines_values(path='./chart_3.jpg',
                               y_range=(40, 120),
                               x_range=(0, -1)):
    """
    Return: Pandas DataFrame 
    """
    img = plt.imread(path)
    img = rgb2gray(img)

    orig_p = get_orig_point(img)

    img = img[:orig_p[0] + 1, orig_p[1]:]

    end_p = get_end_point(img)

    grid_x_idx = get_grid_line_ixd(img, end_p)

    remove_grid_line(img, grid_x_idx)

    img = img[end_p[0]:, :end_p[1]]

    np.place(img, img < 50, 0)
    np.place(img, img > 200, 255)
    np.place(img, (img != 0) & (img != 255), 127)

    start_idx = 1
    while True:
        if np.unique(img[:-2, start_idx]).shape[0] == 1:
            start_idx += 1
        else:
            break

    end_idx = img.shape[1] - 1
    while True:
        if np.unique(img[:-2, end_idx]).shape[0] == 1:
            end_idx -= 1
        else:
            break

    img = img.astype(np.int)

    mid_lines = pd.DataFrame(index=np.arange(img.shape[1]),
                             columns=['x', 0, 127])
    mid_lines['x'] = mid_lines.index
    for v in [0, 127]:
        mid_lines[v] = mid_lines['x'].apply(
            lambda i: get_mid_values(img, i, v))

    mid_lines = mid_lines.iloc[start_idx:end_idx + 1]


    
    f_fill = mid_lines.fillna(method='ffill')
    b_fill = mid_lines.fillna(method='bfill')
    mid_lines = (f_fill + b_fill) // 2
    mid_lines = mid_lines[[0, 127]].fillna(method='ffill')

    y_slope = (y_range[1] - y_range[0]) / img.shape[0]

    mid_lines = img.shape[0] - mid_lines
    mid_lines = mid_lines * y_slope + y_range[0]
    return mid_lines


#################################################################### 以上都是函數的設定下方開始取值


# ### 問題在這邊, '/XObject' 這個物件在六張圖的檔案20180307回診.pdf中不存在, 要怎麼把pdf變成圖片匯出?

f_path = './20180307回診.pdf'     # 檔案名稱(注意路徑), 檔案要跟程式碼擺在一起

pdf_file = open(f_path, 'rb')
reader = PyPDF2.PdfFileReader(pdf_file)

page = reader.getPage(1)  #都在第二頁

if '/XObject' in page['/Resources']:                       # <- 這裡的if不會有東西輸出, 所以沒有img7,8那種東西
    xObject = page['/Resources']['/XObject'].getObject()

    for obj in xObject:
        if xObject[obj]['/Subtype'] == '/Image':
            size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
            data = xObject[obj]._data
            if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                mode = "RGB"
            else:
                mode = "P"

            if '/Filter' in xObject[obj]:
                if xObject[obj]['/Filter'] == '/FlateDecode':
                    data = FlateDecode.decode(
                        data, xObject[obj].get('/DecodeParms'))
                    img = Image.frombytes(mode, size, data.encode())
                    img.save(obj[1:] + ".png")
                elif xObject[obj]['/Filter'] == '/DCTDecode':
                    img = open(obj[1:] + ".jpg", "wb")
                    img.write(data)
                    img.close()
                elif xObject[obj]['/Filter'] == '/JPXDecode':
                    img = open(obj[1:] + ".jp2", "wb")
                    img.write(data)
                    img.close()
                elif xObject[obj]['/Filter'] == '/CCITTFaxDecode':
                    img = open(obj[1:] + ".tiff", "wb")
                    img.write(data)
                    img.close()
            else:
                img = Image.frombytes(mode, size, data)
                img.save(obj[1:] + ".png")
                
