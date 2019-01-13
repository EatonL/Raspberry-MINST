### 软硬件环境：

- [x] 树莓派3B+
- [x] 树莓派官方 CSI 摄像头
- [x] python 3.5
- [x] tensorflow 1.11.0
- [x] opencv 3.4.4

### 环境安装及代码解析

详见以下三篇博客：

[树莓派上玩MINST（一）](https://blog.csdn.net/bazhidao0031/article/details/86377208)

[树莓派上玩MINST（二）](https://blog.csdn.net/bazhidao0031/article/details/86408784)

[树莓派上玩MINST（三）](https://blog.csdn.net/bazhidao0031/article/details/86418806)

### 操作方式

1. 首先按照官方说明依次安装好 CSI摄像头 、耳机（或是扬声器）
2. python3 环境下运行 main.py
3. 在这个界面下：

![](https://img-blog.csdnimg.cn/20190113142637715.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhemhpZGFvMDAzMQ==,size_16,color_FFFFFF,t_70)

将方框对准所选数字，按 '**c**' 截图，多次按下 '**c**' ，后一次截图会覆盖上一次图片，选好后，按 '**q**' 退出选图即可。