#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 第一次使用PyPDF2需要安裝
get_ipython().system('pip install PyPDF2')


# In[2]:


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


# In[3]:


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


# In[87]:


##############需要自己輸入的部分###########################
f_path = './20190306回診.pdf'     # 檔案名稱(注意路徑), 檔案要跟程式碼擺在一起
patient = 'id001'                # 輸出資料給的代號, 可以自己取, 但記得一定要改, 不然舊的輸出會被覆蓋
y_range1 = (0, 60)               # 小圖1的y軸範圍(如果有變動要自己改)
y_range2 = (50, 200)             # 小圖2的y軸範圍(如果有變動要自己改)
y_range3 = (0, 100)              # 小圖3的y軸範圍(如果有變動要自己改)
y_range4 = (40, 120)             # 小圖4的y軸範圍(如果有變動要自己改)
y_range5 = (0, 8)                # 小圖5的y軸範圍(如果有變動要自己改)
base_time = (2018,2,1)           # 請輸入時間軸 "最左邊" 出現的第一個時間, 其年月(例如:2018FEB 就輸入(2018,2), 日期都默認1號, 可不改)
duration = 431                   # 時間全長(天數), 如果有變可以自己更改
#########################################################

pdf_file = open(f_path, 'rb')
reader = PyPDF2.PdfFileReader(pdf_file)

page = reader.getPage(2)  #都在第三頁

if '/XObject' in page['/Resources']:
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
                
# 此步驟會輸出img7.png, img8.jpg兩個圖片, img8.jpg是我們的目標圖片

img = plt.imread('./img8.jpg')

img = rgb2gray(img)

sub_img = img[0:50, :]
plt.imsave(f'timeline.jpg', np.stack([sub_img] * 3, -1))

img = img[183:, :]

c = 1
s = 0
e = 177
sub_img = img[s:e, :]
plt.imsave(f'chart_{c}.jpg', np.stack([sub_img] * 3, -1))

c += 1
s = e + 1
e += 179

sub_img = img[s:e, :]
plt.imsave(f'chart_{c}.jpg', np.stack([sub_img] * 3, -1))

c += 1
s = e + 1
e += 170

sub_img = img[s:e, :]
plt.imsave(f'chart_{c}.jpg', np.stack([sub_img] * 3, -1))

c += 1
s = e + 1
e += 160

sub_img = img[s:e, :]
plt.imsave(f'chart_{c}.jpg', np.stack([sub_img] * 3, -1))

c += 1
s = e + 1
e += 150

sub_img = img[s:e, :]
plt.imsave(f'chart_{c}.jpg', np.stack([sub_img] * 3, -1))

# 產出chart1~chart5分別是我們的五個小圖


# 找時間軸第一個位置
path = './timeline.jpg'
img = plt.imread(path)
img = rgb2gray(img)

# 定位時間軸的位置
temp = 0
for i in range(img.shape[0]):
    if sum(img[i]==0) > temp:
        temp=sum(img[i]==0)
        location = i
# 算出軸起點到第一個月分間的長度(gap)        
for i in range(img.shape[1]):
    if (img[location]==0)[i] == True:
        first_day=i
        break
for i in range(img.shape[1]):
    if (img[location-1]==0)[i] == True:
        first_month=i
        break
gap = first_month - first_day

base = datetime.date(base_time[0], base_time[1], base_time[2])   # 基準時間

# ---------------------------------------- 轉檔圖 1 ----------------------------------------

path = './chart_1.jpg'
img = plt.imread(path)
# 輸入圖片轉灰階img
img = rgb2gray(img)


# 找原點
orig_p = get_orig_point(img)
img = img[:orig_p[0] + 1, orig_p[1]:]

# 找軸長
end_p = get_end_point(img)

# 找虛線位置
grid_x_idx = get_grid_line_ixd(img, end_p)

# 移除虛線
remove_grid_line(img, grid_x_idx)

# 去除軸
img = img[end_p[0]:, :end_p[1]]
img = img[:-1, 1:]

# 轉換一些雜訊到純黑白
np.place(img, img < 50, 0)
np.place(img, img > 200, 255)

# 找出點(減少點數)
r_shape = list(img.shape)

for i in range(2):
    while r_shape[i] % 2 != 0:
        r_shape[i] += 1

heat_map = block_reduce(image=(img == 0).astype(np.int),
                        block_size=(2, 2),
                        func=np.mean)

res = np.where(heat_map > 0.75)

# 這裡有個問題, 因為畫圖是描點的, 所以每個日期(x)會有很多值, 這裡應該只需要保留最大的那個值就可以了, 所以調整一下:

output = pd.DataFrame(np.asarray(res)).transpose().reindex(columns=[1,0])

time = []
index = []
for i in range(output.shape[0]):
    if output[1][i] not in time:
        time.append(output[1][i])
        index.append(i)
        
output = output.loc[index]        
get_time = (output[1]*2).sort_values().reset_index(drop=True)              # 先記下來, 給圖二直接參照
output[1] = round((output[1]*2 - gap)/(end_p[1]) * duration,0)             # 這裡先默認全長都是431天
output[0] = round((r_shape[0] - output[0] * 2) / r_shape[0] * (y_range1[1]-y_range1[0]) + y_range1[0], 4)   # 小數點只保留4位
output.columns = ['time_origin', 'AT/AF total minutes/day']
output = output.sort_values(by='time_origin')
output = output.reset_index(drop=True)
temp=[]
for i in range(len(output['time_origin'])):
    temp.append((base + datetime.timedelta(days = output['time_origin'][i])).strftime("%Y/%m/%d"))
output['time'] = temp
output.to_csv(r'output/'+patient+'_chart1.csv', index=False)



# ---------------------------------------- 轉檔圖 2 ----------------------------------------

path = './chart_2.jpg'
img = plt.imread(path)
# 輸入圖片轉灰階img
img = rgb2gray(img)

# 找原點
orig_p = get_orig_point(img)

img = img[:orig_p[0] + 1, orig_p[1]:]

# 找軸長
end_p = get_end_point(img)

# 找虛線位置
grid_x_idx = get_grid_line_ixd(img, end_p)

# 移除虛線
remove_grid_line(img, grid_x_idx)

# 去除軸
img = img[end_p[0]:, :end_p[1]]
img = img[:-1, 1:]


# img = resize(img, [img.shape[0], 420 * 3]) * 255

np.place(img, img < 50, 0)
np.place(img, img > 200, 255)
# np.place(img, (img != 0) & (img != 255), 127)        
        
value = []
for i in range(len(get_time)):
    for j in range(img.shape[0]):
        if img[j][get_time[i]] == 0:
            value.append(img.shape[0]-j-2)
            break
            
output = pd.DataFrame(list(zip(get_time,value)))
    
output[0] = round((output[0] - gap)/(end_p[1]) * duration,0)
output[1] = round((output[1]) / r_shape[0] * (y_range2[1]-y_range2[0]) + y_range2[0], 4)
output.columns = ['time_origin', 'V. rate during AT/AF(bpm)']
output = output.sort_values(by='time_origin')
output = output.reset_index(drop=True)
temp=[]
for i in range(len(output['time_origin'])):
    temp.append((base + datetime.timedelta(days = output['time_origin'][i])).strftime("%Y/%m/%d"))
output['time'] = temp

output.to_csv(r'output/'+patient+'_chart2.csv', index=False)

# ---------------------------------------- 轉檔圖 3 ----------------------------------------
res = extrat_lines_values('./chart_3.jpg', y_range=y_range3)
res = res.reset_index()
res.columns = ['time_origin', 'Ventricular', 'Artial']
res['time_origin'] = round((res['time_origin'] - gap)/(end_p[1]) * duration,0)
res = res.replace(np.NaN, y_range3[0])                               # 如果有遺失(躺在地上那種)就給最低值

temp=[]
for i in range(len(res['time_origin'])):
    temp.append((base + datetime.timedelta(days = res['time_origin'][i])).strftime("%Y/%m/%d"))
res['time'] = temp

res.to_csv(r'output/'+patient+'_chart3.csv', index=False)


# ---------------------------------------- 轉檔圖 4 ----------------------------------------

res = extrat_lines_values('./chart_4.jpg', y_range=y_range4)
res = res.reset_index()
res.columns = ['time_origin', 'Night', 'Day']
res['time_origin'] = round((res['time_origin'] - gap)/(end_p[1]) * duration,0)
res = res.replace(np.NaN, y_range4[0])                                # 如果有遺失(躺在地上那種)就給最低值

temp=[]
for i in range(len(res['time_origin'])):
    temp.append((base + datetime.timedelta(days = res['time_origin'][i])).strftime("%Y/%m/%d"))
res['time'] = temp

res.to_csv(r'output/'+patient+'_chart4.csv', index=False)


# ---------------------------------------- 轉檔圖 5 ----------------------------------------

res = extrat_lines_values('./chart_5.jpg', y_range=y_range5)
res = res.reset_index()
res.columns = ['time_origin', 'Patient activity hours/day', 'drop']
res['time_origin'] = round((res['time_origin'] - gap)/(end_p[1]) * duration,0)
res = res.replace(np.NaN, y_range5[0])                                # 如果有遺失(躺在地上那種)就給最低值

temp=[]
for i in range(len(res['time_origin'])):
    temp.append((base + datetime.timedelta(days = res['time_origin'][i])).strftime("%Y/%m/%d"))
res['time'] = temp
res = res.drop(columns=['drop'])
res.to_csv(r'output/'+patient+'_chart5.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




