import numpy as np
# numpy中的矩阵运算使用C语言编写，比python自带的字典或者列表更快，
# pandas是numpy的升级版

"""
# numpy基本的属性
array = np.array([[1,2],[3,4]])
print(array)
print(array.ndim) #矩阵维度
print(array.shape) #矩阵行列数
print(array.size) #元素个数
"""

"""
# 创建
print("Create")
a = np.array([1,2,3]) #array关键字创建数组
print(a)

a = np.array([1,2,3], dtype=np.int) #dtype指定数据类型
print(a)
print(a.dtype)
a = np.array([1,2,3], dtype=np.int32)
print(a)
print(a.dtype)
a = np.array([1,2,3], dtype=np.float)
print(a)
print(a.dtype)
a = np.array([1,2,3], dtype=np.float32)
print(a)
print(a.dtype)
"""

""" 
# 特殊创建
print("Special create")
a = np.array([[1,2,3],[4,5,6]]) #直接创建2行3列矩阵
print(a)
a = np.zeros((3,4)) #zeros创建全0矩阵，3行四列
print(a)
a = np.ones((3,4), dtype=np.int) #ones创建全1矩阵，3行4列，可以指定数据类型
print(a)
a = np.empty((3,4)) #empty创建全空数组，每个值接近于0
print(a)
a = np.arange(10,20,2) #arange创建连续数组，10-19的数据步长为2
print(a)
a = np.arange(12).reshape((3,4)) #reshape改变数据的形状，3行4列，0-11
print(a)
a = np.linspace(1,10,20) #创建线段型数据，1-10且分割成20个数据
print(a)
a = np.linspace(1,10,20).reshape((4,5)) #改变形状
print(a)
"""

"""
# numpy运算
a = np.array([10,20,30,40]) # array([10, 20, 30, 40])
b = np.arange(4)            # array([0, 1, 2, 3])
print(a + b)                #矩阵对应位置相加
print(a - b)
print(a * b)                #矩阵对应位置相乘，并不是矩阵乘法
print(a ** 2)               #每个元素求2次方
print(a < 3)                #每个元素进行逻辑判断
print(10 * np.sin(a))       #调用函数对每一项元素进行函数运算

a = np.array([[1,1],[0,1]])
b = np.arange(4).reshape((2,2))
print(np.dot(a,b))          #矩阵标准乘法，行乘列
print(a.dot(b))             #另一种表示形式

a = np.random.random((2,4))
print(a)
print(np.sum(a))            #对整个矩阵进行操作求和、求最小值、求最大值
print(np.min(a))
print(np.max(a))
print(np.sum(a,axis=1))     #axis=1,以行为单位进行操作
print(np.min(a,axis=1))
print(np.max(a,axis=1))
print(np.sum(a,axis=0))     #axis=0,以列为单位进行操作
print(np.min(a,axis=0))
print(np.max(a,axis=0))

print(np.argmin(a))         # 矩阵中最小元素的索引
print(np.argmax(a))         # 矩阵中最大元素的索引

print(np.mean(a))           # 求矩阵平均值
print(np.average(a))
print(a.mean())

print(np.cumsum(a))         # 累加函数，生成的每一项矩阵元素均是从原矩阵首项累加到对应项的元素之和
print(np.diff(a))           # 有差累加函数，计算每一行中后一项与前一项之差
print(np.nonzero(a))        # 将所有非零元素的行与列坐标分割开，重构成两个分别关于行和列的矩阵 
                            # 意思是a中每个非0元素的坐标（i，j），i在关于行的矩阵中，j在关于列的矩阵中
print(np.sort(a))           # 对每一行进行从大到小排序操作，不同行之间不排
print(np.transpose(a))      # 矩阵的转置
print(a.T)
print(np.clip(a,0,9))       # clip(Array,Array_min,Array_max)，顾名思义，Array指的是将要被执行用的矩阵，
                            # 而后面的最小值最大值则用于让函数判断矩阵中元素是否有比最小值小的或者比最大值大的元素，
                            # 并将这些指定的元素转换为最小值或者最大值。
"""

"""
# 索引
a = np.arange(3,15)
print(a[3])                 # 类似读取列表元素的一维索引
a = np.arange(3,15).reshape((3,4))
print(a[2])                 # 类似读取二维列表的一维索引

print(a[1,1])               # 读取矩阵中单个元素的二维索引
print(a[1][1])
print(a[1,1:3])             # 利用":"对一定范围内的元素切片，1:3表示读取1，2

for row in a:               # 逐行打印
    print(row)
for colnum in a.T:          # 逐列打印，必须先转置
    print(colnum)

print(a.flatten())          # 迭代输出，flatten函数将多维矩阵展开成1行的数列
for num in a.flat:          # flat是迭代器，本身是一个对象属性
    print(num)
"""

"""
# 合并
a = np.array([1,1,1])
b = np.array([2,2,2])
c = np.vstack((a,b))        #上下合并，对括号中的两个整体进行对应操作
print(c)
print(a.shape,c.shape)      #a是列表，c是矩阵

d = np.hstack((a,b))        #左右合并
print(d)
print(a.shape,d.shape)

print(a[:,np.newaxis])      #转置操作，对于不是矩阵的属性，无法像前文那样调用矩阵转置函数，
                            #这里可以调用newaxis来实现列表转置
print(a[:,np.newaxis].shape)
print(a[np.newaxis,:])
print(a[np.newaxis,:].shape)

a = np.array([1,1,1])[:,np.newaxis]
b = np.array([2,2,2])[:,np.newaxis]
print(a, b)
c = np.concatenate((a, b, b, a), axis = 0)    #使用concatenate合并多个矩阵或序列
print(c)
c = np.concatenate((a, b, b, a), axis = 1)    #0表示纵向合并，1表示横向合并
print(c)
"""

"""
# 分割
a = np.arange(12).reshape((3, 4))
print(a)
print(np.split(a,2,axis=1)) #axis=1,表示纵向按列等量分成2个列表
print(np.split(a,3,axis=0)) #axis=0,表示横向按行等量分成3个列表

#print(np.split(a, 3, axis=1))#a只有4列，只能等量对分，因此输入以上程序代码后Python就会报错
print(np.array_split(a, 3, axis=1)) #使用array_split将数据不等量分割

print(np.vsplit(a, 3)) #等于 print(np.split(a, 3, axis=0))
print(np.hsplit(a, 2)) #等于 print(np.split(a, 2, axis=1))
"""

"""
# 拷贝赋值
a = np.arange(4)
b = a
b[0] = 5                    #浅拷贝，a、b指向同一地址空间
print(a)

c = a.copy()                #深拷贝，a、c没有关系
c[0] = 6
print(a)
print(c)
"""