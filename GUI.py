import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from datetime import datetime
from time import strftime
from PIL import ImageTk, Image
import imutils
import cv2
from ultralytics import YOLO
import torch
import numpy as np
from collections import deque
from deepsort import *
from numpy import random


source = ''
model_path = ''
class_path = ''
names = None
class_id = []
class_name = []
classes = []
photo = None
youtube_url = ''
data_deque = {}
object_counter = {}


count_frame = 0
dem = 0
density_total = 0
density = deque()
density_final = 0


line2 = np.zeros((2, 2), np.int_)
pts = np.zeros((4, 2), np.int_)
count_line = 0
count_pts = 0


def mouse_callback(event, x, y, flags, param):
    global count_line, count_pts, line2, pts
    if count_line < 2:
        if event == cv2.EVENT_LBUTTONDOWN:
            line2[count_line] = (x, y)
            count_line += 1
    elif count_pts < 4:
        if event == cv2.EVENT_LBUTTONDOWN:
            pts[count_pts] = (x, y)
            count_pts += 1




def compute_color_for_labels(label):
    if label == 2:  # Xe buyt
        color = (0, 149, 255)
    elif label == 3:  # xe hoi
        color = (222, 82, 175)
    elif label == 5:  # Xe may
        color = (0, 204, 255)
    elif label == 7:  # xe tai
        color = (85, 45, 255)
    else:
        color = (255, 255, 255)
    return tuple(color)


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)

    return img


def UI_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
def draw_boxes(img, bbox, object_id, identities=None):
    area_roi = cv2.contourArea(pts)
    mylist = []
    xemay = []
    xehoi = []
    xebuyt = []
    xetai = []
    height, width, _ = img.shape
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

        dist = cv2.pointPolygonTest(pts, (center), False)

        if (dist >= 0):
            #UI_box(box, img, label=label, color=color, line_thickness=2)
            id = int(identities[i]) if identities is not None else 0

            # create new buffer for new object
            if id not in data_deque:
                data_deque[id] = deque(maxlen=64)
            color = compute_color_for_labels(object_id[i])
            obj_name = names[object_id[i]]
            label = '%s' % (obj_name)

            # add center to buffer
            data_deque[id].appendleft(center)
            if len(data_deque[id]) >= 2 :
                if intersect(data_deque[id][0], data_deque[id][1], line2[0], line2[1]):
                    cv2.line(img, line2[0], line2[1], (255, 255, 255), 3)
                    if obj_name not in object_counter:
                        object_counter[obj_name] = 1
                    else:
                        object_counter[obj_name] += 1
            # draw trail
            for i in range(1, len(data_deque[id])):
                # check if on buffer value is none
                if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                    continue
                # generate dynamic thickness of trails
                thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
                # draw trails
                cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
            width_b = abs(box[2] - box[0])
            height_b = abs([box[3]] - box[1])
            area = width_b * height_b
            UI_box(box, img, label=label, color=color, line_thickness=1)
            if (label == "xe may"):
                xemay.append(area)
            if (label == "xe hoi"):
                xehoi.append(area)
            if (label == "xe tai"):
                xetai.append(area)
            if (label == "xe buyt"):
                xebuyt.append(area)


    count = 0


    agv_area_xemay = np.nan_to_num(np.mean(xemay, dtype=np.float64))
    sum_area_xemay = agv_area_xemay * len(xemay)
    agv_area_xehoi = np.nan_to_num(np.mean(xehoi, dtype=np.float64))
    sum_area_xehoi = agv_area_xehoi * len(xehoi)
    agv_area_xetai = np.nan_to_num(np.mean(xetai, dtype=np.float64))
    sum_area_xetai = agv_area_xetai * len(xetai)
    agv_area_xebuyt = np.nan_to_num(np.mean(xebuyt, dtype=np.float64))
    sum_area_xebuyt = agv_area_xebuyt * len(xebuyt)
    density_total = round(((sum_area_xebuyt + sum_area_xetai + sum_area_xemay + sum_area_xehoi) / area_roi), 2)
    return img, count, density_total,object_counter







def handle_dropdown_change(*args):
    global model_path
    if selected_option.get() == "Custom Model..":
        file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pt")])
        if file_path:
            model_path = file_path
        else:
            model_path = ""
    else:
        option_dict = {
            "Yolov8n": "model/yolov8n.pt",
            "Yolov8m": "model/yolov8m.pt",
            "Yolov8l": "model/yolov8l.pt",
            "Yolov8x": "model/yolov8x.pt",
        }
        model_path = option_dict.get(selected_option.get(), "")


def show_lable(file_path):
    capture = cv2.VideoCapture(file_path)
    # Đọc frame đầu tiên
    ret, frame = capture.read()
    if ret:
        # Chuyển đổi frame sang định dạng PIL Image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = imutils.resize(frame, width=1200,height=675)
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)
        label.config(image=photo)
        label.image = photo
    # Giải phóng tài nguyên video
    capture.release()

def handle_dropdown2_change(*args):
    global source,youtube_url
    selected_value = selected_option2.get()
    if selected_value == "Video":
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if file_path:
            # Handle logic when a video file is selected
                # Đọc video
            show_lable(file_path)
            source = file_path
        else:
            # Handle logic when no video file is selected
            source = ""

    elif selected_value == "Camera":
        # Handle logic for Camera option
        source = 1
        show_lable(source)

    elif selected_value == "Stream Video":
        stream_url = simpledialog.askstring("Stream Video", "Enter the video stream URL:")
        if stream_url:
            # Handle logic when a video stream URL is entered
            source = stream_url
            show_lable(source)



def handle_dropdown3_change(*args):
    global selected_class, position, class_id, class_name, label_jpg
    selected_class = selected_option3.get()
    position = classes.index(selected_class)

    if position in class_id:
        class_id.remove(position)
        class_name.remove(selected_class)
    else:
        class_id.append(position)
        class_name.append(selected_class)

    if 'label_jpg' in globals() and label_jpg is not None:
        label_jpg.destroy()

    label_jpg = tk.Label(text="Class : " + str(class_name))
    label_jpg.place(x=350, y=180)

def read_classes_from_file(file_path):
    global names
    with open(file_path, 'r') as file:
        lines = file.readlines()
        classes = [line.strip() for line in lines]
        names = classes
    return classes,names


def load_classes():
    global class_path
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("Names Files", "*.names")])
    class_path = file_path
    if file_path:
        global classes
        classes,_ = read_classes_from_file(file_path)
        menu = dropdown3['menu']
        menu.delete(0, 'end')
        for item in classes:
            menu.add_command(label=item, command=lambda value=item: selected_option3.set(value))
def drawROI():
    global count_line, count_pts, line2, pts
    capture = cv2.VideoCapture(source)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        if count_line == 2:
            cv2.line(frame, line2[0], line2[1], (255, 0, 0), 2)
        if count_pts == 4:
            mask_ROI = np.zeros_like(frame)
            cv2.fillPoly(mask_ROI, [pts], (128, 255, 0))
            cv2.polylines(mask_ROI, [pts], True, (255, 255, 128), 2)
            frame = cv2.addWeighted(frame, 1, mask_ROI, 0.2, 0)
        cv2.imshow('Results video', frame)
        cv2.setMouseCallback('Results video', mouse_callback)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
def show_video():
    global count_frame,dem,density_total,density,density_final
    if source == '':
        messagebox.showerror("Lỗi", "Bạn chưa chọn nguồn.")
        return
    elif model_path == '':
        messagebox.showerror("Lỗi", "Bạn chưa chọn model.")
        return
    elif class_path == '':
        messagebox.showerror("Lỗi", "Bạn chưa chọn classes.")
        return
    try:
        cap = cv2.VideoCapture(source)
        print(source)
        model= YOLO(model_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while True:
            value_scale = scale.get()
            value_scale2 = scale2.get()
            print(value_scale,value_scale2)
            ret, frame = cap.read()
            mask_ROI = np.zeros_like(frame)
            cv2.fillPoly(mask_ROI, [pts], (128, 255, 0))
            cv2.polylines(mask_ROI, [pts], True, (255, 255, 128), 2)
            cv2.line(frame, line2[0], line2[1], (255, 0, 112), 3)
            frame = cv2.addWeighted(frame, 1, mask_ROI, 0.2, 0)
            results = model.predict(frame, conf=value_scale, iou=value_scale2, device=0, classes=class_id)
            a = results[0].boxes.cuda()
            boxes = a.cpu().numpy()
            xywh = boxes.xywh
            confs = boxes.conf
            oids = []
            for cls in boxes.cls:
                oid = int(cls)
                oids.append(oid)
            xywh_tensor = torch.Tensor(xywh)
            confs_tensor = torch.Tensor(confs)
            outputs = deepsort.update(xywh_tensor, confs_tensor, oids, frame)
            if (len(outputs)) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -2]
                object_id = outputs[:, -1]
                offset = (0, 0)
                count_frame += 1

                for i, box in enumerate(bbox_xyxy):
                    x1, y1, x2, y2 = [int(i) for i in box]
                    x1 += offset[0]
                    x2 += offset[0]
                    y1 += offset[1]
                    y2 += offset[1]
                    frame, count, density_total,obj = draw_boxes(frame, bbox_xyxy, object_id, identities)
                for idx, (key, value) in enumerate(obj.items()):
                    cnt_str1 = str(key) + "  :  " + str(value)
                    count += value
                    # Xóa label_count cũ
                    if 'label_count' in globals() and label_count is not None:
                        label_count.destroy()

                    # Tạo label_count mới
                    label_count = tk.Label(text=str(cnt_str1), fg="blue", bg='white',
                                           font=("times new roman", 24, 'bold'))
                    label_count.place(x=1600, y=180 + (idx * 80))
                    label_count.configure(background='#F8F8FF', foreground='#00008B')

                    # Xóa label_sum cũ
                    if 'label_sum' in globals() and label_sum is not None:
                        label_sum.destroy()

                    # Tạo label_sum mới
                    label_sum = tk.Label(text="Tổng:  " + str(count), fg="blue", bg='white',
                                         font=("times new roman", 24, 'bold'))
                    label_sum.place(x=1600, y=180 + ((idx + 1) * 80))
                    label_sum.configure(background='#F8F8FF', foreground='#00008B')

                    # Xóa label_density cũ



            density.append(density_total)
            if (count_frame % 150 == 0):
                density_avg = sum(density) / len(density)
                density_final = round((density_avg * 100), 2)
                density.clear()


            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = imutils.resize(frame, width=1200)
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            label.config(image=photo)
            label.image = photo

            if 'density_lable' in globals() and density_lable is not None:
                density_lable.destroy()
            density_lable = tk.Label(text="Mat do : " + str(density_final) + " %", fg="blue", bg='white',
                                     font=("times new roman", 26, 'bold'))
            density_lable.place(x=350, y=900)
            density_lable.configure(background='#F8F8FF', foreground='#00008B')
            root.update()

    except Exception as e:
        # Xử lý lỗi ở đây, ví dụ: hiển thị thông báo lỗi
        return





def ngaythangnam():
    rnow = strftime('%d/%m/%Y')
    current_time = datetime.now().strftime("%H:%M:%S")
    lb = tk.Label(root, text="", fg="blue", bg='white', font=("times new roman", 50, 'bold'))
    lb.place(x=1500, y=20)
    lb.configure(text=current_time, background='#F8F8FF', foreground='#00008B')
    lb2 = tk.Label(root, text="", fg="blue", bg='white', font=("times new roman", 14, 'bold'))
    lb2.place(x=1600, y=100)
    lb2.configure(text=rnow, background='#F8F8FF', foreground='#00008B')
    lb.after(1000, ngaythangnam)

def help():
    messagebox.showinfo("Help","Bước 1: Chọn model (Yolov8x, Yolov8n, Yolov8S, ... hoặc custom model\nBước 2: Chọn source (video từ máy tính, camera...)\nBước 3 : Vẽ line, ROI (2 click đầu để vẽ line ngang đường, 4 click sau để vẽ vùng ROI.\nBước 4: Detection")



root = tk.Tk()
root.title('ĐỒ ÁN TỔNG HỢP')
root.configure(background='#F8F8FF')

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")

bg = tk.PhotoImage(file='logo2.png')
bg = bg.subsample(6)

bgpt = tk.Label(root, image=bg)
bgpt.configure(background='#F8F8FF', foreground='#FF0000')
bgpt.place(x=0, y=0)

label = tk.Label(root)
label.pack()
label.place(x=350, y=200)

options = ["Chọn model...", "Yolov8n", "Yolov8m", "Yolov8l", "Yolov8x", "Custom Model.."]
selected_option = tk.StringVar(root)
selected_option.set(options[0])
selected_option.trace("w", handle_dropdown_change)

selected_option2 = tk.StringVar(root)
options2 = ["Chọn source ...", "Video", "Camera", "Stream Video"]
selected_option2.set(options2[0])
selected_option2.trace("w", handle_dropdown2_change)

selected_option3 = tk.StringVar(root)
selected_option3.set("Chọn classes")
selected_option3.trace("w", handle_dropdown3_change)

dropdown1 = tk.OptionMenu(root, selected_option, *options)
dropdown1.pack()
dropdown1.place(x=160, y=200)

dropdown2 = tk.OptionMenu(root, selected_option2, *options2)
dropdown2.pack()
dropdown2.place(x=160, y=280)

dropdown3 = tk.OptionMenu(root, selected_option3, ())
dropdown3.pack()
dropdown3.place(x=180, y=600)

scale = tk.Scale(root, from_=0, to=1, resolution=0.05, length=250, orient=tk.HORIZONTAL)
scale.set(0.5)
scale.place(x=40, y=400)

scale2 = tk.Scale(root, from_=0, to=1, resolution=0.05, length=250, orient=tk.HORIZONTAL)
scale2.set(0.7)
scale2.place(x=40, y=520)

btn4 = tk.Button(root, text="Vẽ ROI, line", command=drawROI)
btn4.configure(background='#364156', foreground='white', font=('times new roman', 15, 'bold'), activeforeground='#FFFF63')
btn4.pack(side=tk.BOTTOM, padx=50, pady=50)
btn4.configure(width=20)
btn4.place(x=40, y=680)


btn5 = tk.Button(root, text="Detection", command=show_video)
btn5.configure(background='#364156', foreground='white', font=('times new roman', 15, 'bold'), activeforeground='#FFFF63')
btn5.pack(side=tk.BOTTOM, padx=50, pady=50)
btn5.configure(width=20)
btn5.place(x=40, y=760)

btn6 = tk.Button(root, text="Load classes", command=load_classes)
btn6.configure(background='#364156', foreground='white', font=('times new roman', 15, 'bold'), activeforeground='#FFFF63')
btn6.pack(side=tk.BOTTOM, padx=50, pady=50)
btn6.place(x=40, y=600)


btn7 = tk.Button(root, text="Trợ giúp",command=help)
btn7.configure(background='#364156', foreground='white', font=('times new roman', 15, 'bold'), activeforeground='#FFFF63')
btn7.pack(side=tk.BOTTOM, padx=50, pady=50)
btn7.configure(width=20)
btn7.place(x=40, y=860)

btn8 = tk.Button(root, text="Thoát")
btn8.configure(background='#364156', foreground='white', font=('times new roman', 15, 'bold'), activeforeground='#696969')
btn8.pack(side=tk.BOTTOM, padx=100, pady=100)
btn8.configure(width=20)
btn8.place(x=40, y=940)

heading = tk.Label(root, text="Trường Đại học Công Nghệ Tp.HCM", pady=10, font=('times new roman', 20, 'bold'))
heading.configure(background='#F8F8FF', foreground='#00008B')
heading.pack()
heading.place(x=750, y=20)

heading1 = tk.Label(root, text="Nghiên cứu khoa học sinh viên 2023", pady=10, font=('times new roman', 20, 'bold'))
heading1.configure(background='#F8F8FF', foreground='#00008B')
heading1.pack()
heading1.place(x=750, y=65)

heading2 = tk.Label(root, text="Đề tài: Hệ thống đo lường mật độ giao thông", pady=10, font=('times new roman', 16, 'bold'))
heading2.configure(background='#F8F8FF', foreground='#00008B')
heading2.pack()
heading2.place(x=750, y=115)

heading_model = tk.Label(root, text="Chọn model:", font=('times new roman', 14, 'bold'))
heading_model.configure(background='#F8F8FF', foreground='#00008B')
heading_model.pack()
heading_model.place(x=40, y=200)

heading_model = tk.Label(root, text="Chọn source:", font=('times new roman', 14, 'bold'))
heading_model.configure(background='#F8F8FF', foreground='#00008B')
heading_model.pack()
heading_model.place(x=40, y=280)

heading_conf = tk.Label(root, text="Conf:", font=('times new roman', 14, 'bold italic'))
heading_conf.configure(background='#F8F8FF', foreground='#00008B')
heading_conf.pack()
heading_conf.place(x=40, y=360)

heading_iou = tk.Label(root, text="IoU:", font=('times new roman', 14, 'bold italic'))
heading_iou.configure(background='#F8F8FF', foreground='#00008B')
heading_iou.pack()
heading_iou.place(x=40, y=480)

ngaythangnam()
root.mainloop()