
# Base Images 
## 从天池基础镜像构建(from的base img 根据自己的需要更换，建议使用天池open list镜像链接：https://tianchi.aliyun.com/forum/postDetail?postId=67720) 
FROM ac2-registry.cn-hangzhou.cr.aliyuncs.com/ac2/pytorch:2.5.1.6-cuda12.1.1-py310-alinux3.2104
#registry.cn-hangzhou.aliyuncs.com/sais-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04


# 创建所需的目录结构
RUN mkdir -p /app /saisresult /saisdata /feature

## 把当前文件夹里的文件构建到镜像的根目录下,并设置为默认工作目录 
ADD . /app

# 设置工作目录
WORKDIR /app

# 安装必要的依赖（根据您的实际需求修改）
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /app/requirements.txt

# RUN python /app/download.py

# 确保run.sh有可执行权限
RUN chmod +x /app/run.sh

# 镜像启动后执行run.sh(直接执行，不需要sh前缀)
CMD ["sh","/app/run.sh"]

