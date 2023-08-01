# python 运行:

目前仅支持pq模式,需要先在主目录下进行编译, 编译方式参考主页README.md,运行命令如下:

```bash
python test_pq.py --device_id 0 --mode_set s --nq 10 --d 256 --M 32 --nbits 8 --ntotal 400 --topk 1
```
python版运行，本质是对C++实现的接口进行封装和调用，上述运行参数和"tests/run_test_cnindex_pq.sh"中一致
