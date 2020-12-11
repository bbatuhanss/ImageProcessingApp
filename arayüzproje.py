# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 00:25:52 2020

@author: han_b
"""
import skimage.io as io 
from skimage import color,morphology,exposure,filters
from skimage.transform import rotate,resize,swirl,rescale,warp_polar 
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
from skimage.filters import gaussian
import cv2 
from PIL import ImageTk, Image
from tkinter import filedialog
import os
top = Tk()
top.geometry("400x400")
top.title("Image Processing")
top.resizable(width=True, height=True)
frameAna=Frame(top)
frameAna.pack(side=TOP)
x='C:/Users/han_b/Desktop/x.jpg'
resim= io.imread('C:/Users/han_b/Desktop/x.jpg')
img=color.rgb2gray(resim)
filterimage=resim
def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

"filters"
def gaussianFilter():
    global filterimage
    imgFilter3=filters.gaussian(img, sigma=1, output=None, mode='nearest', cval=0, multichannel=None, preserve_range=False, truncate=4.0)
    filterimage=imgFilter3
    io.imshow(imgFilter3)
    io.show() 
   
def sobelFilter():
    global filterimage
    imgFilter1 = filters.sobel(img)
    filterimage=imgFilter1
    io.imshow(imgFilter1)
    io.show() 
def unsharp_maskFilter():
    global filterimage
    imgFilter2=filters.unsharp_mask(img, radius=1.0, amount=1.0)
    filterimage=imgFilter2
    io.imshow(imgFilter2)
    io.show() 
def satoFilter():
    global filterimage
    imgFilter4=filters.sato(img, sigmas=range(1, 10, 2), black_ridges=True, mode=None, cval=0)
    filterimage=imgFilter4
    io.imshow(imgFilter4)
    io.show() 
def prewittFilter():
    global filterimage
    imgFilter5=filters.prewitt(img,axis=1)
    filterimage=imgFilter5
    io.imshow(imgFilter5)
    io.show() 
def medianFilter():
    global filterimage
    imgFilter6=filters.median(img, selem= morphology.disk(15))
    filterimage=imgFilter6
    io.imshow(imgFilter6)
    io.show() 
def meijeringFilter():
    global filterimage
    imgFilter7=filters.meijering(img, sigmas=range(1, 10, 2), alpha=None, black_ridges=True, mode='reflect', cval=0)
    filterimage=imgFilter7
    io.imshow(imgFilter7)
    io.show()
def hessianFilter():
    global filterimage
    imgFilter8=filters.hessian(img, sigmas=range(1, 10, 2), scale_range=None, scale_step=None, alpha=0.5, beta=0.5, gamma=15, black_ridges=True, mode=None, cval=0)
    filterimage=imgFilter8
    io.imshow(imgFilter8)
    io.show()
def frangiFilter():
    global filterimage
    imgFilter9=filters.frangi(img, sigmas=range(1, 10, 2), scale_range=None, scale_step=None, alpha=0.5, beta=0.5, gamma=15, black_ridges=True, mode='reflect', cval=0)
    filterimage=imgFilter9
    io.imshow(imgFilter9)
    io.show()
def laplaceFilter(): 
    global filterimage      
    imgFilter10= filters.laplace(img)
    filterimage=imgFilter10
    io.imshow(imgFilter10) 
    io.show()
    
"Uzaysal Dönüşüm"
def rotateUzaysal():  
    global filterimage      
    rotateimg= rotate(img,75)
    filterimage=rotateimg
    io.imshow(rotateimg)
    io.show() 
def resizeUzaysal():   
    global filterimage     
    resizeimg=resize(img, (300, 300))
    filterimage=resizeimg
    io.imshow(resizeimg)
    io.show()
def swirlUzaysal():  
    global filterimage      
    swrilimg=swirl(img, rotation=0, strength=25, radius=800)
    filterimage=swrilimg
    io.imshow(swrilimg)
    io.show() 
def rescaleUzaysal():
    global filterimage      
    rescaleimg=rescale(img, scale=0.2, order=None, mode='reflect', cval=0, clip=True, preserve_range=False, multichannel=False, anti_aliasing=None, anti_aliasing_sigma=None)
    filterimage=rescaleimg
    io.imshow(rescaleimg)
    io.show()
def warp_polarUzaysal():  
    global filterimage      
    warped=warp_polar(img ,scaling='log')
    filterimage=warped
    io.imshow(warped)
    io.show()
    
    
"Morfolojik İşlemler "
def  opening():
    global filterimage
    opened = morphology.opening(img,selem=None, out=None)
    filterimage=opened
    io.imshow(opened)
    io.show()
def  closing():
    global filterimage
    closed = morphology.closing(img,selem=None, out=None)
    filterimage=closed
    io.imshow(closed)
    io.show()

def  erosion():
    global filterimage
    eroded =morphology.erosion(img, selem=None, out=None, shift_x=False, shift_y=False)
    filterimage=eroded
    io.imshow(eroded)
    io.show()
            
def  dilation():
    global filterimage
    dilated =morphology.dilation(img,selem=None, out=None, shift_x=False, shift_y=False)
    filterimage=dilated
    io.imshow(dilated)
    io.show()

def  black_tophat():
    global filterimage
    b_tophat =morphology.black_tophat(img,selem=None, out=None)
    filterimage=b_tophat
    io.imshow(b_tophat)
    io.show()

def  diameterOpening():
    global filterimage
    sk =morphology.diameter_opening(img, diameter_threshold=8, connectivity=1, parent=None, tree_traverser=None)
    filterimage=sk
    io.imshow(sk)
    io.show()
"değiştir"      
def  diameterClosing():
    global filterimage
    hull1 =morphology.diameter_closing(img, diameter_threshold=8, connectivity=1, parent=None, tree_traverser=None)
    filterimage=hull1
    io.imshow(hull1)
    io.show()
def  white_tophat(): 
    global filterimage
    kernel = morphology.disk(5)
    img_white =morphology.white_tophat(img, kernel)
    filterimage=img_white
    io.imshow(img_white)
    io.show()

def  areaopening():
    global filterimage
    wate=morphology.area_opening(img, area_threshold=64, connectivity=1, parent=None, tree_traverser=None)
    filterimage=wate
    io.imshow(wate)
    io.show()
        
def  areaclosing():
    global filterimage
    tree=morphology.area_closing(img, area_threshold=64, connectivity=1, parent=None, tree_traverser=None)
    filterimage=tree
    io.imshow(tree)
    io.show()

def SAveImge():
    global top
    top.destroy()
    top=Tk()
    top.geometry("500x300")
    top.title("Save")
    framefilter=Frame(top)
    framefilter.pack()
    label1 = Label(framefilter,  text="Kaydedeceğiniz resmin adını giriniz(ismin sonuna '.jpg'  koyunuz!'')",font=("Helvetica", 10))
    label1.pack(side=TOP)
    ımagesave = Label(top, text = "Lütfen Kaydedilecek ismi girin:").place(x = 20,y = 47)  
    esave = Entry(top)
    esave.pack(pady=30)
 
    def kaydet():
        io.imsave(esave.get(),filterimage)
    button=Button(top, text ="Save", command = kaydet,height = 2, width = 30,bg = "orange").pack(side = TOP)
    label2 = Label(top, height=1)
    label2.pack(side=TOP)                                       
    Button(top, text ="AnaSayfa", command = AnaSayfa,height = 2, width = 30,bg = "blue").pack(side = TOP)

def KendiFiltrem():
    global filterimage
    eroded =morphology.erosion(resim, selem=None, out=None, shift_x=False, shift_y=False)
    imgFilter6=filters.median(eroded)
    imgFilter2=filters.unsharp_mask(imgFilter6, radius=1.0, amount=1.0)
    rotateimg= rotate(imgFilter2,15)
    filterimage=rotateimg
    io.imshow(rotateimg)
    io.show() 


    
def videoFilter():
    okunanVideo = cv2.VideoCapture('C:/Users/han_b/Desktop/z.mp4')

    while (True):
        ret, videoGoruntu = okunanVideo.read()
        resize=cv2.resize(videoGoruntu,(800,600) ,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        griTon=cv2.cvtColor(resize,cv2.COLOR_RGB2GRAY)
        edge=cv2.Canny(griTon,50,60)
        cv2.imshow("Video Okuma Islemi", edge)
        if cv2.waitKey(10) & 0xFF == ord('x'):
            break
    okunanVideo.release()
    cv2.destroyAllWindows()

def ActiveCounter():
    s = np.linspace(0, 2*np.pi, 300)
    x = 475 + 150*np.cos(s)
    y = 260 + 150*np.sin(s)
    init = np.array([x, y]).T
    cat = active_contour(gaussian(resim, 3),init, alpha=0.015, beta=10, gamma=0.001)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(resim)
    ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
    ax.plot(cat[:, 0], cat[:, 1], '-b', lw=3)
    
    plt.show()
    
def morfolojik():  

    global top
    top.destroy()
    top=Tk()
    top.geometry("600x800")
    top.title("Morfolojik İşlemler ")
    label1 = Label(top,  text="Morfolojik İşlemler ",font=("Helvetica", 25))
    label1.pack(side=TOP)
    label2 = Label(top, height=1)
    label2.pack(side=TOP)
    framemorfo=Frame(top)
    framemorfo.pack()
    label1 = Label(top,  text="Orginal Resim",font=("Helvetica", 10))
    label1.pack(side=TOP)
    goster = Image.open(x)
    goster = goster.resize((250, 250), Image.ANTIALIAS)
    goster = ImageTk.PhotoImage(goster)
    panel = Label(top, image=goster)
    panel.image = goster
    panel.pack()
    
    Button(framemorfo, text ="Opening", command = opening, height = 2, width = 18, bg = "light blue").pack()
    Button(framemorfo, text ="Closing", command = closing ,height = 2, width = 18,bg = "light blue").pack()
    Button(framemorfo, text ="Erosion", command = erosion,height = 2, width = 18,bg = "light blue").pack()
    Button(framemorfo, text ="Dilation", command = dilation,height = 2, width = 18,bg = "light blue").pack()
    Button(framemorfo, text ="Blacktophat", command = black_tophat,height = 2, width = 18,bg = "light blue").pack()
    Button(framemorfo, text ="Diameter Opening", command = diameterOpening,height = 2, width = 18,bg = "light blue").pack()
    Button(framemorfo, text ="Diameter Closing", command = diameterClosing,height = 2, width = 18,bg = "light blue").pack()
    Button(framemorfo, text ="Whitetophat", command = white_tophat,height = 2, width = 18,bg = "light blue").pack()
    Button(framemorfo, text ="Area Opening", command = areaopening,height = 2, width = 18,bg = "light blue").pack()
    Button(framemorfo, text ="Area closing", command = areaclosing,height = 2, width = 18,bg = "light blue").pack()
    Button(framemorfo, text ="Save", command = SAveImge,height = 2, width = 25,bg = "blue").pack()
    Button(framemorfo, text ="AnaSayfa", command = AnaSayfa,height = 2, width = 25,bg = "blue").pack()

def UzaysalScreen():
    global top
    top.destroy()
    top=Tk()
    top.geometry("500x600")
    top.title("Uzaysal Dönüşüm")
    label1 = Label(top,  text="Uzaysal Dönüşüm ",font=("Helvetica", 25))
    label1.pack(side=TOP)
    label2 = Label(top, height=1)
    label2.pack(side=TOP)
    frameUzay=Frame(top)
    frameUzay.pack()
    label1 = Label(top,  text="Orginal Resim",font=("Helvetica", 10))
    label1.pack(side=TOP)
    goster = Image.open(x)
    goster = goster.resize((250, 250), Image.ANTIALIAS)
    goster = ImageTk.PhotoImage(goster)
    panel = Label(top, image=goster)
    panel.image = goster
    panel.pack()
    
    Button(frameUzay, text ="Rotate", command = rotateUzaysal, height = 2, width = 18, bg = "light blue").pack()
    Button(frameUzay, text ="Resize", command = resizeUzaysal ,height = 2, width = 18,bg = "light blue").pack()
    Button(frameUzay, text ="Swirl", command = swirlUzaysal,height = 2, width = 18,bg = "light blue").pack()
    Button(frameUzay, text ="Rescale", command = rescaleUzaysal,height = 2, width = 18,bg = "light blue").pack()
    Button(frameUzay, text =" Warppolar", command = warp_polarUzaysal,height = 2, width = 18,bg = "light blue").pack()
    Button(frameUzay, text ="Save", command = SAveImge,height = 2, width = 25,bg = "blue").pack()
    Button(frameUzay, text ="AnaSayfa", command = AnaSayfa,height = 2, width = 25,bg = "blue").pack()


def FilterScreen():
    global top
    top.destroy()
    top=Tk()
    top.geometry("600x800")
    top.title("Filters")
    label1 = Label(top,  text="Filters",font=("Helvetica", 25))
    label1.pack(side=TOP)
    label2 = Label(top, height=1)
    label2.pack(side=TOP)
    frameFilter=Frame(top)
    frameFilter.pack()
    label1 = Label(top,  text="Orginal Resim",font=("Helvetica", 10))
    label1.pack(side=TOP)
    goster = Image.open(x)
    goster = goster.resize((250, 250), Image.ANTIALIAS)
    goster = ImageTk.PhotoImage(goster)
    panel = Label(top, image=goster)
    panel.image = goster
    panel.pack()
    panel.pack(side = "bottom", fill = "both", expand = "yes")
    Button(frameFilter, text ="Gaussian", command = gaussianFilter, height = 2, width = 18, bg = "light blue").pack()
    Button(frameFilter, text ="Sobel", command = sobelFilter ,height = 2, width = 18,bg = "light blue").pack()
    Button(frameFilter, text ="Unsharpmask", command = unsharp_maskFilter,height = 2, width = 18,bg = "light blue").pack()
    Button(frameFilter, text ="Sato", command = satoFilter,height = 2, width = 18,bg = "light blue").pack()
    Button(frameFilter, text ="Prewitt", command = prewittFilter,height = 2, width = 18,bg = "light blue").pack()
    Button(frameFilter, text ="Median", command = medianFilter,height = 2, width = 18,bg = "light blue").pack()
    Button(frameFilter, text ="Meijering", command = meijeringFilter,height = 2, width = 18,bg = "light blue").pack()
    Button(frameFilter, text ="Hessian", command = hessianFilter,height = 2, width = 18,bg = "light blue").pack()
    Button(frameFilter, text ="frangi", command = frangiFilter,height = 2, width = 18,bg = "light blue").pack()
    Button(frameFilter, text ="laplace", command = laplaceFilter,height = 2, width = 18,bg = "light blue").pack()
    Button(frameFilter, text ="Save", command = SAveImge,height = 2, width = 25,bg = "blue").pack()
    Button(frameFilter, text ="AnaSayfa", command = AnaSayfa,height = 2, width = 25,bg = "blue").pack()

def histogra():
    global filterimage
    histoimg=exposure.equalize_hist(img)
    filterimage=histoimg
    io.imshow(histoimg)
    io.show()

def Yogunluk():
    global top
    top.destroy()
    top=Tk()
    frameyogunluk=Frame(top)
    frameyogunluk.pack()
    top.geometry("500x500")
    top.title("Yoğunluk Dönüşüm İşlemleri")
    label1 = Label(frameyogunluk,  text="Yoğunluk Dönüşüm İşlemleri",font=("Helvetica", 10))
    label1.pack(side=TOP)
    label2 = Label(top, height=1)
    label2.pack(side=TOP)
    
    goster = Image.open(x)
    goster = goster.resize((250, 250), Image.ANTIALIAS)
    goster = ImageTk.PhotoImage(goster)
    panel = Label(top, image=goster)
    panel.image = goster
    panel.pack()
    panel.pack(side = "bottom", fill = "both", expand = "yes")
    ment1=StringVar()
    ment2=StringVar()
    deger1 = Label(top, text = "Yogunluk Deger1").place(x = 50,y = 55)  
    deger2 = Label(top, text = "Yogunluk Deger2").place(x = 50, y = 90)  
    e1 = Entry(top)
    e1.pack(pady=10)
    e2 = Entry(top)
    e2.pack(pady=10)
    label1 = Label(top,  text="Orginal Resim",font=("Helvetica", 10)).place(x = 200, y = 202) 
    
    def tamamla():
        global filterimage
        rescaleintensity=exposure.rescale_intensity(img,  out_range=(int(e1.get()), int(e2.get())))
        filterimage=rescaleintensity
        io.imshow(rescaleintensity)
        io.show()
    button=Button(top, text ="Yoğunluklu Göster", command = tamamla,height = 2, width = 30,bg = "light blue").pack(side = TOP)                                       
    Button(top, text ="Save", command = SAveImge,height = 2, width = 30,bg = "orange").pack()
    Button(top, text ="AnaSayfa", command = AnaSayfa,height = 2, width = 30,bg = "blue").pack(side = TOP)
       
      
def AnaSayfa():
    global top
    top.destroy()
    top=Tk()
    top.geometry("500x700")
    top.title("Image Processing")
    frameAna2=Frame(top)
    frameAna2.pack()
    label3 = Label(frameAna2,  text="Görüntü İşlemleri",font=("Helvetica", 25))
    label3.pack(side=TOP)
    label2 = Label(frameAna2, height=1)
    label2.pack(side=TOP) 
    label1 = Label(top,  text="Orginal Resim",font=("Helvetica", 10))
    label1.pack(side=TOP)
    goster = Image.open(x)
    goster = goster.resize((250, 250), Image.ANTIALIAS)
    goster = ImageTk.PhotoImage(goster)
    panel = Label(top, image=goster)
    panel.image = goster
    panel.pack()
    
    Button(frameAna2, text ="Filitre İşlemleri", command = FilterScreen,height = 2, width = 22,bg = "light blue").pack()
    Button(frameAna2, text ="Histogram İşlemleri", command = histogra,height = 2, width = 22,bg = "light blue").pack()
    Button(frameAna2, text ="Uzaysal Dönüşüm İşlemleri", command = UzaysalScreen,height = 2, width = 22,bg = "light blue").pack()
    Button(frameAna2, text ="Yoğunluk Dönüşümü İşlemleri", command = Yogunluk,height = 2, width = 22,bg = "light blue").pack()
    Button(frameAna2, text ="Morfolojik İşlemleri", command = morfolojik,height = 2, width = 22,bg = "light blue").pack()
    Button(frameAna2, text ="Active Contour İşlemi", command = ActiveCounter,height = 2, width = 22,bg = "light blue").pack()
    Button(frameAna2, text ="Special Filter", command = KendiFiltrem,height = 2, width = 22,bg = "light blue").pack()
    Button(frameAna2, text ="Save", command = SAveImge,height = 2, width = 25,bg = "blue").pack()
    Button(frameAna2, text ="Resim Değiştir", command = EnDısAnasayfa,height = 2, width = 30,bg = "blue").pack()

def open_img():
    global resim
    global img
    global x
    x = openfn()
    resim = io.imread(x)
    img_gray=color.rgb2gray(resim)
    img=img_gray
    global top
    top.destroy()
    top=Tk()
    top.geometry("500x700")
    top.title("Image Processing")
    frameAna3=Frame(top)
    frameAna3.pack()
    label3 = Label(frameAna3,  text="Görüntü İşlemleri",font=("Helvetica", 25))
    label3.pack(side=TOP)
    label2 = Label(frameAna3, height=1)
    label2.pack(side=TOP) 
    label1 = Label(top,  text="Orginal Resim",font=("Helvetica", 10))
    label1.pack(side=TOP)
    goster = Image.open(x)
    goster = goster.resize((250, 250), Image.ANTIALIAS)
    goster = ImageTk.PhotoImage(goster)
    panel = Label(top, image=goster)
    panel.image = goster
    panel.pack()
    
    Button(frameAna3, text ="Filitre İşlemleri", command = FilterScreen,height = 2, width = 22,bg = "light blue").pack()
    Button(frameAna3, text ="Histogram İşlemleri", command = histogra,height = 2, width = 22 ,bg= "light blue").pack()
    Button(frameAna3, text ="Uzaysal Dönüşüm İşlemleri", command = UzaysalScreen,height = 2, width = 22,bg = "light blue").pack()
    Button(frameAna3, text ="Yoğunluk Dönüşümü İşlemleri", command = Yogunluk,height = 2, width = 22,bg = "light blue").pack()
    Button(frameAna3, text ="Morfolojik İşlemleri", command = morfolojik,height = 2, width = 22,bg = "light blue").pack()
    Button(frameAna3, text ="Active Contour İşlemi", command = ActiveCounter,height = 2, width = 22,bg = "light blue").pack()
    Button(frameAna3, text ="Special Filter", command = KendiFiltrem,height = 2, width = 22,bg = "light blue").pack()
    Button(frameAna3, text ="Save", command = SAveImge,height = 2, width = 25,bg = "blue").pack()
    Button(frameAna3, text ="Resim Değiştir", command = EnDısAnasayfa,height = 2, width = 30,bg = "blue").pack()


def EnDısAnasayfa():
    global top
    top.destroy()
    top=Tk()
    top.geometry("400x400")
    top.title("Welcome Image Processing Program")
    label1 = Label(top,  text="Image Processing İşlemler İçin Resim Seçiniz",font=("Helvetica", 10))
    label1.pack(side=TOP)
    label2 = Label(top, height=1)
    label2.pack(side=TOP)
    frameAna4=Frame(top)
    frameAna4.pack()
    Button(frameAna4, text ="Resim Seç", command = open_img,height = 2, width = 18,bg = "light blue").pack()
    Button(frameAna4, text ="Video İşleme  İşlemi", command = videoFilter,height = 2, width = 18,bg = "light blue").pack()

Label(frameAna,  text="Image Processing İşlemler İçin Resim Seçiniz",font=("Helvetica", 12),width = 50).pack() 

Label(frameAna, height=5).pack()
Button(frameAna, text ="Video İşleme  İşlemi", command = videoFilter,height = 2, width = 18,bg = "light blue").pack(side=BOTTOM)
Button(frameAna, text ="Resim Seç", command = open_img,height = 2, width = 18,bg = "light blue").pack(side=BOTTOM)

        

    
    

top.mainloop()
