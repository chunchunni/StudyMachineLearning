import pandas as pd
import numpy as np

"""
# 用 python 的列表和字典来作比较, 那么可以说 Numpy 是列表形式的，没有数值标签，而 Pandas 就是字典形式。
# Pandas是基于Numpy构建的，让Numpy为中心的应用变得更加简单。

# pandas中主要两个数据结构：Series和DataFrame。

# Series的字符串表现形式为：索引在左边，值在右边。
# 由于我们没有为数据指定索引。
# 于是会自动创建一个0到N-1（N为长度）的整数型索引
s = pd.Series([1,3,6,np.nan,44,1])
print(s)

# DataFrame是一个表格型的数据结构，它包含有一组有序的列，每列可以是不同的值类型（数值，字符串，布尔值等）。
# DataFrame既有行索引也有列索引， 它可以被看做由Series组成的大字典。
dates = pd.date_range('20210101',periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
# randn是从标准正态中返回一个或多个样本值
print(df)

print(df['b'])              # 打印df中b对应列

df1 = pd.DataFrame(np.arange(12).reshape((3,4)))    # 不指定标签的数据，会默认索引起始为0
print(df1)

df2 = pd.DataFrame({'A' : 1.,
                    'B' : pd.Timestamp('20210101'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo'})
# 指定列索引，然后按列赋值
print(df2)

print(df2.dtypes)           # 查看每列对应的数据类型
print(df2.index)            # 查看每列对应的序号
print(df2.columns)          # 查看每列的索引
print(df2.values)           # 只查看所有的值
print(df2.describe)         # 对数据进行总结
print(df2.T)                # 翻转数据
print(df2.sort_index(axis= 1, ascending= False))    # 对index排序后输出
print(df2.sort_values(by='B'))                      # 对数据值进行排序输出
"""

"""
# 选择数据
dates = pd.date_range('20210101',periods=6)
df = pd.DataFrame(np.arange(24).reshape(6,4),index=dates,columns=['a','b','c','d'])
print(df)

print(df.a)                 # 选取索引对应列的数据
print(df['a'])
print(df[0:3])              # 输出多行，左闭右开
print(df['20210101':'20210104']) # 按行输出索引，左右都包括

print(df.loc['20210101'])   # 通过标签选择一行的数据
print(df.loc[:,['a','b']])  # 通过选择某行或者所有行（:代表所有行）然后选其中某一列或几列数据

print(df.iloc[1,1])         # 通过位置选择在不同情况下所需要的数据,1 1表示第2行第2列
print(df.iloc[3:5,1:3])     # 连续选多行多列中的元素
print(df.iloc[[1,3,5],1:3]) # 跨行选元素

print(df[df.a>8])           # 约束某个条件来选择当前数据
"""

"""
# 设置值
dates = pd.date_range('20210101', periods = 6)
df = pd.DataFrame(np.arange(24).reshape((6,4)), index = dates, columns = ['a','b','c','d'])
print(df)

df.iloc[2,2] = 1111         # 利用索引或者标签确定需要修改值的位置
df.loc['20210101','b'] = 2222
print(df)

df.b[df.a > 4] = 0          # 根据条件设置
print(df)

df['f'] = np.nan            # 对整列做批处理，加一列f
print(df)

# 使用Series序列加上数据，但是长度必须对其
df['e'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20210101',periods=6)) 
print(df)
"""

"""
# 处理丢失的数据
dates = pd.date_range("20210101", periods = 6)
df = pd.DataFrame(np.arange(24).reshape(6,4), index=dates, columns=['a','b','c','d'])
print(df)

df.iloc[0,1] = np.nan       # 将矩阵中的部分值设置为NaN数据，表示未定义或不可表示的值。
df.iloc[1,2] = np.nan
print(df)

print(df.dropna(            # dropna函数去掉含NaN的行或列
    axis = 0,               # 0: 对行进行操作; 1: 对列进行操作
    how = 'any'             # 'any': 只要存在 NaN 就 drop 掉; 'all': 必须全部是 NaN 才 drop 
    ))

print(df.fillna(value = 0)) # fillna函数将 NaN 的值用其他值代替, 比如代替成 0

print(df.isnull())          # 判断是否有缺失数据 NaN, 为 True 表示缺失数据
"""

"""
# 文件导入导出
data = pd.read_csv('./data/student.csv')    # 读取csv文件
print(data)

data.to_pickle('./data/student.pickle')     # 将数据存为pickle
"""

"""
# concat合并
df1 = pd.DataFrame(np.zeros((3,4)), columns = ['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4)), columns = ['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns = ['a','b','c','d'])

# axis控制合并方向
df4 = pd.concat([df1,df2,df3], axis = 0)    # axis表示合并方向，未设定任何参数时，函数默认axis=0 纵向合并
print(df4)

# append添加行数据
print(df1.append(df2, ignore_index=True))   # 将df2合并到df1的下面，以及重置index，并打印出结果
print(df1.append([df2, df3], ignore_index=True)) # 合并多个df，将df2与df3合并至df1的下面，以及重置index
s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])
print(df1.append(s1, ignore_index=True))    # df1添加有一行数据。合并series，将s1合并至df1，以及重置index，并打印出结果

# join控制合并
# join='outer'为预设值，因此未设定任何参数时，函数默认join='outer'。
# 此方式是依照column来做纵向合并，有相同的column（列索引相同）上下合并在一起。
# 其他独自的column各自成列，原本没有值的位置皆以NaN填充
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])
# df1和df2各有一个索引不同
df5 = pd.concat([df1,df2], axis=0, join='outer')
print(df5)
# inner 只有相同的column合并在一起，其他的会被抛弃
df6 = pd.concat([df1,df2], axis=0, join='inner')
print(df6)
"""
