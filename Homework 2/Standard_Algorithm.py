import cv2


class PanoramaStitching:
    """
    全景拼接
    """
    def __init__(self):
        images = [cv2.imread(r'./data1/112_1298.JPG'), cv2.imread('./data1/112_1299.JPG'), cv2.imread('./data1/112_1300.JPG')]
        '''
        这段代码使用了列表推导式，目的是对读取的图像进行缩放处理。我们可以逐步分解并详细解释这一行代码：
        self.images = [...]：这部分表示将处理后的图像存储在对象的self.images属性中。self指代对象的实例，这意味着这些图像是特定于该对象的属性。
        [cv2.resize(...) for img in images]：这是列表推导式，它是在对images列表中的每个图像（img）进行操作。for img in images的含义是遍历images列表中的每个图像。
        cv2.resize(src=img, dsize=None, fx=0.5, fy=0.5)：这是OpenCV库中的一个函数，用于调整图像的大小。
        src=img：指定要调整大小的源图像，即当前遍历到的图像img。
        dsize=None：表示不指定目标图像的大小。此参数为None意味着目标大小将由缩放因子fx和fy决定。
        fx=0.5和fy=0.5：表示在水平方向和垂直方向上都将图像缩小到原始大小的50%。具体来说，fx和fy分别是横向和纵向的缩放因子。
        结合以上分析，这行代码的主要作用是将读取的图像列表中的每个图像都缩小到原来的一半，并将它们存储在对象的self.images属性中。
        '''
        self.images = [cv2.resize(src=img, dsize=None, fx=0.5, fy=0.5) for img in images]

    def run(self):

        stitcher = cv2.Stitcher().create()
        status, pano = stitcher.stitch(self.images)
        '''
        创建拼接器实例：
        stitcher = cv2.Stitcher().create()
        CopyInsert
        cv2.Stitcher()：这是OpenCV库中的一个类，用于进行图像拼接。拼接器的主要作用是将一系列的输入图像拼接成一个全景图。
        .create()：该方法用于创建一个拼接器对象，准备执行拼接操作。此时，stitcher变量就引用了一个已初始化的拼接器实例。
        执行拼接操作：
        
        status, pano = stitcher.stitch(self.images)
        CopyInsert
        stitcher.stitch(self.images)：调用拼接器的stitch方法，对之前已准备好的图像列表self.images进行拼接。此方法会返回两个值：
        status：拼接操作的状态，其值可以是多个预定义的常量。具体来说，cv2.STITCHER_OK表示拼接成功，其他值则表示拼接过程中出错。
        pano：拼接生成的全景图像，当拼接成功时，该变量将保存拼接后的结果图像。
        '''
        if status == cv2.STITCHER_OK:
            cv2.imshow('pano', pano)
            cv2.waitKey(0)
        else:
            print('无法拼接为全景图')


if __name__ == '__main__':
    ps = PanoramaStitching()
    ps.run()


