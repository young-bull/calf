import os
from time import sleep
import pandas as pd

# k,epochs,learning_rate,weight_decay,units,dropout
def build_params(params_dic, params_file='./params_file.csv'):
    """
    params_dic: key:字符串，参数名称；value: list，参数值
    """

    # 排列组合
    def expand(mulcoldf, sigcoldf):
        """
        mulcolddf: multi columns dataframe
        sigcoldf:  single column dataframe
        """
        if mulcoldf.empty:
            return sigcoldf

        r = pd.DataFrame()
        for x in sigcoldf.values:
            s = mulcoldf.copy()
            s[sigcoldf.columns[0]] = x[0]
            r = pd.concat([r,s])
        return r

    p = pd.DataFrame()
    for k,v in params_dic.items():
        p = expand(p, pd.DataFrame({k:v}))
    p = p.reset_index(drop=True)
    p.to_csv(params_file, index=False)
    return p

def get_params(params_file='./params_file.csv'):
    return pd.read_csv(params_file)

import threading
mutex = threading.Lock()

def get_param(num, params_file='params_file.csv'):
    """
    num 从1开始
    """
    params_file = os.path.abspath(params_file)
    if not os.path.isfile(params_file):
        return
    params = pd.read_csv(params_file) # 默认会去掉第一行
    return None, params[(num-1):num]

def get_todo_param(params_file='params_file.csv'):
    '''
    2个param文件
     1. xxxnet_params: 存放所有待测参数，有header
     2. got_xxxnet_params: 目前已经取到的行，此数字为 params 中的 index 编号
    
    return: pd.DataFrame（不能返回 pd.Series，因为Series只有1个dtype）
    '''
    params_file = os.path.abspath(params_file)
    if not os.path.isfile(params_file):
        return
    params = pd.read_csv(params_file) # 默认会去掉第一行
    #print(params)
    i = 0
    params_got_file = os.path.dirname(params_file)+"/got_"+os.path.basename(params_file)
    # r：只读、r+：读写 —— 文件不存在则报错，存在则会将位置指针置首
    # w：只写、w+：读写 —— 文件不存在创建之，存在则会将文件内容清零
    # a：只读、a+：读写 —— 文件不存在创建之，存在则会将位置指针置尾
    # seek 是移动读取指针，所以必须在 read 之前，read 到内存的数据不受seek影响
    mod_str = 'r+'
    if not os.path.exists(params_got_file):
        mod_str = 'w+'
    
    #p = pd.DataFrame(columns=params.columns.values)
    mutex.acquire()
    with open(params_got_file, mod_str) as f:
        l = f.readlines()
        if len(l) != 0:
            i = int(l[0].rstrip())
        i += 1
        print("will get %d(%d) set params" % (i, params.shape[0]))
        if i > (params.shape[0]):
            mutex.release()
            return i,None
        f.seek(0)
        f.write(str(i))
        df = params[(i-1):i]
        #print(df)
    mutex.release()
    return i,df


# Test Code
if __name__ == '__main__':

    # 无序字典
    params_dic = {
        'num_epochs' : [ 1,2,50,80 ],
        'learning_rate' : [ 0.1,0.01,0.05 ],
        'weight_decay' : [ 5e-4, 5e-1],
        'lr_period' : [ 80, ],
        'lr_decay' : [ 0.1, ],
    }
    print(params_dic.items())

    # 有序字典
    import collections
    params_coldic = collections.OrderedDict()
    params_coldic['num_epochs'] = [ 1,2,3,50,200,2000,5000]
    params_coldic['learning_rate'] = [ 0.1,0.01,0.05,0.005 ]
    params_coldic['weight_decay'] = [ 5e-4, 5e-1]
    params_coldic['lr_period'] = [ 80, 100 ]
    params_coldic['lr_decay'] = [ 0.1, ]
    print(params_coldic.items())

    params = build_params(params_coldic, params_file='./test_params_file.csv')
    print(params)

    for i in range(100):
        p = get_todo_param(params_file)
        print(p)
        sleep(0.1)