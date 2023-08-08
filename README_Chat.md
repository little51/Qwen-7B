# 通义千问实践

## 1、安装环境

```bash
# clone 代码
git clone https://github.com/little51/Qwen-7B --depth=1
cd Qwen-7B
# 建python3.9虚拟环境
conda create -n qwen python=3.9 -y
conda activate qwen
# 安装依赖包
pip3 install -r requirements.txt  -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
pip3 install fastapi -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
pip3 install uvicorn -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
```

## 2、下载模型

```bash
python model_download.py --repo_id Qwen/Qwen-7B-Chat --mirror
```

## 3、测试基本对话功能

```bash
# 8bit 量化，需要10G显存
CUDA_VISIBLE_DEVICES=0 python test.py
```

## 4、流式API测试

```bash
# 运行api_stream服务
CUDA_VISIBLE_DEVICES=0 nohup python -u api_stream.py > qwen.log  2>&1 &
# 多次发POST请求，直到返回的response中包含[stop]后停止调用
curl -X POST "http://127.0.0.1:8008/stream" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "你是谁？", "history": []}'
```
