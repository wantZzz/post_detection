# movenet與姿態分析技術文件

## 專案目標與實作方向
*“讓姿勢診斷的技術分享給所有人，讓社會擁有更高的身體自覺，從而減少醫療資源浪費。”*

以AI透過站姿靜態分析（正面側面背面）、坐姿動態分析（正面側面）判斷出某種傾向，使使用者提高對於自身體的異常的警覺
根據結果與資料趨勢給與一點非處方運動治療，並標註無法改善請洽詢醫生，且可以給與醫生由程式取得的相關資料。

## 資料處理流程

![](https://i.imgur.com/aYus5Gd.png "資料處理流程")

## 工程進度與日程

| 項目名稱                | 開始日期  | 狀態   | 結束日期  | 備註                                         |
| ----------------------- | --------- | ------ | --------- | -------------------------------------------- |
| 測試 MoveNet 效果與研究 | 2023/1/16 | 已完成 | 2023/1/19 |                                              |
| 開發測試應用程式        | 2023/1/18 | 已完成 | 2023/1/22 | 錄影改拍照記錄                               |
| 演算三維關節位置        | 2023/1/27 | 已完成 | 2023/1/30 |                                              |
| 推算肌肉收縮            | 2023/1/31 | 進行中 |           |                                              |
| 手掌朝向分類(手臂用)     | 2023/1/31 | 進行中 |           | 推算肘部肌肉收縮時，肘部旋轉須納入考慮 |
| 計算模型準確度          |           | 待辦   |           |                                              |
| 正常資料與異常資料輸入  |           | 待辦   |           |                                              |

## 資料輸入

該項要求設備需傳回以下數據:

1. 5至10秒鐘且總影格數不低於200的mp4、wav檔案
2. 拍攝當下鏡頭焦距的數據
3. 拍攝當下加速度、陀螺儀的數據[^同步與誤差]

為將MoveNet定位出的平面二維關節定位透過演算轉換至三維空間定位，而在錄製中不免會搖晃到鏡頭造成關節定位移動，因此需要鏡頭焦距與加速度器數據進行成像分析與修正

考量到功能需要盡可能簡易操作，5至10秒鐘為一個理想的操作時間，且大多數手機僅管性能不好也至少能錄製一條30fps、10秒鐘以內的影片
經演算轉換的三維關節定位將用以分析每條肌肉肌頭與肌尾之間的距離，用於判斷肌肉的拉伸程度

### 開發測試應用程式

測試應用程式選擇以 android 為基底進行開發，為節省開發時間我使用 app inventor 2 加上 ProCamera[^pro-camera] 與自己寫的擴充套件進行資料收集

![](https://i.imgur.com/hQ1xBt3.jpg)

由於需要在拍攝當下記錄加速度、陀螺儀數據以便後續處理資料，但以 app inventor 2 原生的錄影方式無法達到同時記錄的要求，由於在網路上找不到達成要求錄影擴充套件而先以拍照擴充套件將圖像數據以圖片紀錄，以求後續概念驗證完畢後改以錄影方式記錄

## MoveNet定位關節部位

以下為使用 MoveNet lightning 與 thunder 檢測關節在圖片上的位置的程式，該部分展示了如何將影片經 cv2 取出單幀畫面並在調整畫面長寬交由模型進行運算
```python
  #檢測關節在圖片上的位置
  """
  #lightning version
  image2 = cv2.resize(image, (192,192), interpolation=cv2.INTER_AREA)

  img = image2.copy()
  img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
  input_image = tf.cast(img, dtype=tf.float32)
  """

  #thunder version
  image2 = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)

  img = image2.copy()
  img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256,256)
  input_image = tf.cast(img, dtype=tf.float32)
  

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
  interpreter.invoke()
  keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
```

程式內`img`為傳入影片中單一幀畫面，將mp4由opencv傳入、調整並由 Numpy 將資料轉換成float32，以長寬256x256的數據以符合模型的傳入`interpreter.get_input_details`

經 MoveNet 演算的結果將從`interpreter.get_output_details`取出，以下為使用 thunder 版本並用 cv2 繪製的結果

![](https://i.imgur.com/r9ES4H0.png)


## 將關節二維位置演算至三維空間

僅憑一張照片上關節位置無法正確測量出各關節的距離，但是能透過大量的資料數據演算出關節在三維空間的趨近位置
關鍵是需要大量照片以不同角度對一個或多個點觀察，透過簡單的連線與並去除離群值則能得到趨近於現實由各個關節組成、相近比例的連線骨架

### 圖像感測器與三維空間的演算

![](https://i.imgur.com/ubbOiA9.jpg "不同幀的影像感測器對應在三維空間的位置與平面法向量")

上圖紅綠兩個平面分別對應不同幀的影像感測器位置及法向量，使用者在手持手機拍攝多張照片或拍攝影片時不免有些許的搖晃，對應在三維空間會產生不同法向量的平面


```python
#演算平面的位置
now_acceleration_sensor_value = np.array(frame[0])
now_velocity = np.array(frame[1])/fps

##旋轉角度
velocity_x, velocity_y, velocity_z = now_velocity
##上一張照片的加速度
acceleration_x, acceleration_y, acceleration_z = last_acceleration

#velocity_x: 以手機長邊中點連線為軸旋轉，手機上緣向螢幕方向旋轉為正向
#velocity_y: 以手機短邊中點連線為軸旋轉，手機左緣向螢幕方向旋轉為正向
#velocity_z: 以過手機螢幕中點垂直於螢幕垂線為軸旋轉，手機上緣向手機左緣方向旋轉為正向

acc_x = now_acceleration_sensor_value[0] - (acceleration_x*np.cos(velocity_y*np.pi/180) + acceleration_z*np.sin(velocity_y*np.pi/-180) + acceleration_y*np.sin(velocity_z*np.pi/180) + acceleration_x*np.cos(velocity_z*np.pi/180))
acc_y = now_acceleration_sensor_value[1] - (acceleration_y*np.cos(velocity_x*np.pi/180) + acceleration_z*np.sin(velocity_x*np.pi/180) + acceleration_y*np.cos(velocity_z*np.pi/180) + acceleration_y*np.sin(velocity_z*np.pi/-180))
acc_z = now_acceleration_sensor_value[2] - (acceleration_y*np.sin(velocity_x*np.pi/-180) + acceleration_z*np.cos(velocity_x*np.pi/180) + acceleration_x*np.sin(velocity_y*np.pi/180) + acceleration_z*np.cos(velocity_y*np.pi/180))
  
#演算平面的法向量

N, H, W = last_frame_vector

N_stand = N / ((N[0]**2 + N[1]**2 + N[2]**2)**0.5)
H_stand = H / (sensor_longer_side/2)
W_stand = W / (sensor_shorter_side/2)

X_s = np.sin(velocity_x*np.pi/180)
X_c = np.cos(velocity_x*np.pi/180)
Y_s = np.sin(velocity_y*np.pi/180)
Y_c = np.cos(velocity_y*np.pi/180)
Z_s = np.sin(velocity_z*np.pi/180)
Z_c = np.cos(velocity_z*np.pi/180)

##三維旋轉矩陣
R = np.array([[   Y_c * Z_c - X_s * Y_s * Z_s , - X_c * Z_s , Y_s * Z_c + X_s * Y_c * Z_s],
              [   Y_c * Z_s + X_s * Y_s * Z_c ,   X_c * Z_c , Y_s * Z_s - X_s * Y_c * Z_c],
              [ - X_c * Y_s                   ,   X_s       , X_c * Y_c                  ]])
  
N_unprocessed = R.dot(np.array([0.0, 0.0, 1.0]))
H_unprocessed = R.dot(np.array([0.0, 1.0, 0.0]))
W_unprocessed = R.dot(np.array([1.0, 0.0, 0.0]))

#紀錄結果

last_frame_center += W_unprocessed*acc_x*((1/fps)**2) + H_unprocessed*acc_y*((1/fps)**2) + N_unprocessed*acc_z*((1/fps)**2)

H_unprocessed *= sensor_longer_side/2
W_unprocessed *= sensor_shorter_side/2

last_frame_vector = np.array([(N_unprocessed[0]*W_stand + N_unprocessed[1]*H_stand + N_unprocessed[2]*N_stand),
                              (H_unprocessed[0]*W_stand + H_unprocessed[1]*H_stand + H_unprocessed[2]*N_stand),
                              (W_unprocessed[0]*W_stand + W_unprocessed[1]*H_stand + W_unprocessed[2]*N_stand)])
```

雖然這些搖晃移動幅度較小，但符合`不同角度對一個或多個點觀察`的要求，而在後續演算成三維空間時也能更好的挑出因 MoveNet 定位失誤所造成的錯誤、將離群值挑出

### 將關節二維位置演算至三維平面

![](https://i.imgur.com/EXy7dO4.jpg "理想情況下三維空間關節位置推算情形")

將 MoveNet 對每幀進行二維關節定位在這些平面上，這些關節定位可視為人體關節在影像感測器上經焦點映射到平面的點，將二維關節定位對應在平面上並根據當下焦點數據與焦點連線得一條直線方程式

```python
#測算平面在位移與角度翻轉後中心點與左上角空間位置與向量關係
longer_side_vector = last_frame_vector[1]*-2
shorter_side_vector = last_frame_vector[2]*2
focal_point = last_frame_center+(last_frame_vector[0]*focal)
last_frame_left_top_corner = last_frame_center + last_frame_vector[1] - last_frame_vector[2]

#演算關節在平面上的位置與與焦點連線向量
count = 0
for kp in keypoints_with_scores:
  ky, kx, kp_conf = kp
  if kp_conf > confidence_threshold:
    ky = longer_side_vector*ky
    kx = shorter_side_vector*kx
    keypoint_in_flat = last_frame_left_top_corner + ky + kx
    line_vector = focal_point - keypoint_in_flat
      
    try:
      joint_connection[str(count)].append([keypoint_in_flat, line_vector])
    except:
      joint_connection[str(count)] = [[keypoint_in_flat, line_vector]]
    
  count += 1
```
以下為用 Matplotlib 呈現前後兩幀位移、旋轉情形與單幀畫面上關節、相機焦點位置(模型使用 MoveNet thunder 版本，單位:毫米)

![](https://i.imgur.com/HPeTpYb.png "前後兩幀位移、旋轉情形與單幀畫面上關節、相機焦點位置")

### 演算關節三維空間

該直線方程式將與其他幀代出的直線方程式將能用於推算出關節在三維空間的位置:

理想情況下，重復該步驟在不同平面上所得的直線應會相交與一點，而該交點即為該關節在三維空間的位置
實際情況下，應取不同平面上所得的直線最近點，即與多條直線最近的點，而該點即為該關節在三維空間的為止
`intersection_of_multi_lines`取自 Felix Zhang 所寫的文章內容[^多條直線最近點]

```python
#取關節連線焦點後所得的所有直線焦點或最近點
def intersection_of_multi_lines(strt_points, directions):
  n, dim = strt_points.shape

  G_left = np.tile(np.eye(dim), (n, 1))
  G_right = np.zeros((dim * n, n))

  for i in range(n):
    G_right[i * dim:(i + 1) * dim, i] = -directions[i, :]

  G = np.concatenate([G_left, G_right], axis=1)
  d = strt_points.reshape((-1, 1))

  m = np.linalg.inv(np.dot(G.T, G)).dot(G.T).dot(d)

  return m

for joint_id in joint_connection.keys():
  joint_lines = joint_connection[joint_id]
  joint_lines_count = len(joint_lines)

  strt_point = np.zeros((joint_lines_count, 3))
  directions = np.zeros((joint_lines_count, 3))

  count = 0
  for joint_line in joint_lines:
    strt_point[count, :] = joint_line[0]
    directions[count, :] = joint_line[1]
    count += 1

  inters = intersection_of_multi_lines(strt_point, directions)

  Three_dimensional_joint__space_coordinates[joint_id] = [inters[0], inters[1], inters[2]]
```

以下為用 Matplotlib 呈現實際計算後結果(模型使用 MoveNet thunder 版本，單位:公尺)

![](https://i.imgur.com/I5oKGxd.png "關節在三維空間的位置")


## 結合模型推算肌肉收縮

將手部、腿部、軀幹關節位置相互連結即能建構出一副人體骨架，而這副骨架後續在判斷肌肉伸縮時會作為基準點，將肌肉伸縮的情形推算出來

![](https://i.imgur.com/is8WthI.jpg)
![](https://i.imgur.com/9AQFiNB.jpg)

上圖使用現有資料將骨骼平均直徑與截面模擬出骨骼的連接情形，模擬出手臂肱骨(深藍色)、橈骨與尺骨(淺藍色)的大致連接情形[^模型特點]

該骨架亦可在日後取得更精細骨骼資料後套入位移與向量後得到更精細、精確的資料，現階段以構造簡單的形體得到初步的結果

![](https://i.imgur.com/2oiA4ID.jpg)
![](https://i.imgur.com/FEZ2a1S.jpg)

根據資料肌肉的兩端肌頭與肌尾會分別附著於骨架和或其他肌肉上，而兩端也是肌肉唯一不會因肌肉伸縮而位置有大幅變動的部分，兩者連線的長度適合用於判斷肌肉異常與否
上圖為模擬肱肌的連接情形，肌頭位於肱骨正面遠處3/2的位置，肌尾則位於尺骨粗隆(橈骨與尺骨中間)

從資料中將特定肌肉上肌頭肌尾附著在骨架和其他肌肉的點連線，並與已分析的正常與異常資料的相互比對即可將異常的肌肉辨別出來

[^同步與誤差]:加速度、陀螺儀鏡頭焦距的數據需同時紀錄，以利後續將圖像感測器與三維空間的演算
[^pro-camera]:https://community.appinventor.mit.edu/t/pro-camera-the-pro-custom-camera/25353
[^多條直線最近點]:https://zhuanlan.zhihu.com/p/146190385?utm_id=0
[^模型特點]:考慮到橈骨與尺骨之間距離相近且有將肘部旋轉納入考慮，模型將橈骨與尺骨合併為一個上下兩面扭轉一定角度的橢圓柱
