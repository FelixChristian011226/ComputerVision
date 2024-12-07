# Homework1-Photometric Stereo Report

## 一、实验目的

实验的目的是通过实现光度立体（Photometric Stereo）算法，利用多张不同光照方向的图像来估计物体表面的法向量和反射率，并根据其来重新渲染指定光照方向下的图像。

- **法向量计算**：根据朗博模型的公式$I=\rho\overrightarrow{n}\cdot\overrightarrow{l}$，$\rho$是表面的反射率，$\overrightarrow{n}$是法向量，$\overrightarrow{l}$是光照方向。根据课程介绍的算法，当至少提供三张已知光照方向的图像时，$\rho$和$\overrightarrow{l}$都可以唯一确定。
- **阴影高光处理**：阴影和高光打破了线性的朗博模型，一个简单的解决方案是对每个像素的所有观测值排序，丢弃某一百分比的最亮和最暗像素，以去除阴影和高光。



## 二、实验原理

<img src="https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/0dade61b941065d1325153ebd11cd867.png" alt="Lambert" style="zoom: 33%;" />

根据朗博模型（这是一个纯漫反射的模型），对于每一个光线反射点，有个固定的反射率$\rho_0$，而反射光线$L_o$的强度仅仅与反射率$\rho_0$和与入射光线角度$\theta_i$（与法向量的夹角）有关。公式如下：
$$
L_o=L_i\rho_0\cos\theta_i=L_i\rho_0\boldsymbol{n}\cdot\boldsymbol{l}
$$
<img src="https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/a0fd7c0140bfcafeb008f056358914a7.png" alt="image-20241207132846933" style="zoom: 33%;" />

当我们拥有不同光照方向的图像时，就可以利用朗博模型对每个像素点进行单独的分析了。当我们有三个光照角度的图像时，可以以矩阵形式获得朗博模型下的等式：
$$
\begin{pmatrix}
I_1 \\
I_2 \\
I_3
\end{pmatrix}=
\begin{pmatrix}
\boldsymbol{l}_1^T \\
\boldsymbol{l}_2^T \\
\boldsymbol{l}_3^T
\end{pmatrix}\mathrm{\rho}\boldsymbol{n}
$$
由于反射光强$I_i$和入射光角度$\boldsymbol{l}_i$都是已知的，所以通过三个光照角度的数据，就可以通过简单的矩阵求逆获得反射率$\mathrm{\rho}$和法向量$\boldsymbol{n}$。

知道反射率和法向量的基础上，再对指定光照角度的图像渲染就很容易了，直接根据朗博模型求每个像素的光照即可。



## 三、实验内容

在这次实验给出的代码框架里，需要自己实现的基本上只有`myPMS.m`文件。

<img src="https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/53166fad82ee50e249ecc6a5f518cde9.png" alt="image-20241207134101820" style="zoom:50%;" />

首先，对原本给出的函数声明进行了改动，因为最后要求输出的三幅图像，但是给出的框架里面又只输出了法向量，所以把反射率图像和重渲染图像也加到函数返回里了。然后函数输入新增了一个`shadow_removal_percentage`，它表示我在处理阴影高光是要舍弃的最亮最暗的像素值的百分比（比如它的值如果是20，那我要舍弃最暗的20%和最亮的20%）。

在数据预处理阶段，用`N`、`albedo`、`re_rendered_img`分别表示法向量图、反射率图和重渲染图。它们都是三通道的。`I`用来暂存三个通道的光照强度值。

<img src="https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/1d0863310ac9b5c8a1ee62cfa2fa5735.png" alt="image-20241207134531010" style="zoom:50%;" />

通过简单的循环遍历，先是对原图像经过了`mask`，通过遮罩把主体提取出来，防止背景的影响。然后在RGB的每个通道分别除以给出的`light_intensity`，获取每个通道真实的光照强度`I`。

<img src="https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/12820516d1dd2c93dc4760f293d792fc.png" alt="image-20241207134953399" style="zoom:50%;" />

然后再遍历每一个像素点，对所有图像的光照强度进行排序，舍弃最亮和最暗的给定百分比的部分，存放到`I_col_filtered`里面。根据公式$I=\rho\overrightarrow{n}\cdot\overrightarrow{l}$，可以先反推$\rho$，简单的通过光照方向`s_filtered`和光照强度`I_col_filtered`作最小二乘法，即可获得每个像素点的反射率和法向量的积`A`。对`A`求norm即可获得反射率的值，然后对`A`除以这个反射率即可获得单位法向量。

<img src="https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/20fdbf89aed0546975b7f8d49893a762.png" alt="image-20241207140409970" style="zoom:50%;" />

有以上的信息，重渲染就很简单了。直接遍历对每个像素通过朗博模型的公式套上去就可以了。但是由于作业要求里好像没有要求入射光的RGB分量，所以就默认都为1了。

至此，光度立体法的主体函数就完成了，以下是对Baseline进行的小修改：

<img src="https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/0ace3623602bd317629dd0b1f47f7056.png" alt="image-20241207140539892" style="zoom:50%;" />

首先函数调用这块，按照我改之后的输入输出来的，这里为处理阴影高光舍弃的百分比为20，即丢弃最暗的20%和最亮的20%。为了最后显示的重渲染图像更美观，这里还将它进行了归一化。

然后为了更好找输出文件，把它们都输出到`output`文件夹里了，如果老师需要测试一下代码，还烦请保留一下文件夹结构，谢谢~



## 四、实验结果

|        | Normal Map                                                   | Albedo Map                                                   | Re-rendered Picture                                          |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| bear   | <img src="https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/5ae7f5df5e480e32313eeadfc218a1ec.png" alt="bearPNG_Normal"  /> | <img src="https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/11126f4fba79a877396cbb649b85eae2.png" alt="bearPNG_Albedo"  /> | <img src="https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/b1e038833411195869dccfc94e9311c9.png" alt="bearPNG_ReRendered"  /> |
| buddha | ![buddhaPNG_Normal](https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/77db0ed558a8954d54ed24bd548b61ec.png) | ![buddhaPNG_Albedo](https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/624f4c0c73b765a0b7725128394ead80.png) | ![buddhaPNG_ReRendered](https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/551786fccd59bfbe22ee559df3473d6d.png) |
| cat    | ![catPNG_Normal](https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/1ca2781d392d4258eee056952fa7ac6e.png) | ![catPNG_Albedo](https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/63a725360b18d28c825f364ab06f1c4f.png) | ![catPNG_ReRendered](https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/e897663e32bde8e338e11397699d2ed1.png) |
| pot    | ![potPNG_Normal](https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/198d54d72124b26b50beea61be1a33a2.png) | ![potPNG_Albedo](https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/a6ed46b80000be1e8402a751e14fdbe4.png) | ![potPNG_ReRendered](https://pub-70fb49a5419e4021a1be66effc7bcf9e.r2.dev/2024/12/4bf4ac3fb69d12f957b4f42600ebb596.png) |



## 五、实验总结

这次实验，严格意义上讲是我第一次接触Matlab，本来想用C++完成，但是给的框架都是Matlab，也不敢自己乱搞了。感觉最大的困难还是线性代数学得不扎实，对矩阵运算很不熟练，再加上不会Matlab，很容易搞出来error。其实看懂课上讲的方法后，感觉思路还是挺简单的，但是一上手就废了。对我而言，这个算法最难的部分就是最小二乘法那里。因为3幅图像下，完全可以通过求逆获得，但更多幅图像只能通过最小二乘法来最优拟合，而惭愧的是，我基本忘光光了，更别说怎么在Matlab使用它。更尴尬的是，最开始以为是灰度图，做到后面才发现data里面有个值光照强度根本没用上，这才发现是三通道的。

最后生成的效果看起来还挺像那回事。关于什么样的数据效果更好，我感觉粗糙物体的运算结果最贴近真实，因为粗糙物体的表面反射主要为漫反射，更符合朗博模型，而光滑物体一般会有镜面反射，用朗博模型运算出来肯定偏差较大。关于这个算法的改进空间，其实处理阴影高光那块，我很随意的直接丢弃了20%，事实上应该可以根据需求来调整这个值实现更优。当前通过最小二乘法来计算法线和反射率，或许也可以改成更优的方法，比如基于正则化的优化方法等。为了提高当前运算的效率，或许有改进措施可以同时对多像素进行并行运算，以提高效率。



