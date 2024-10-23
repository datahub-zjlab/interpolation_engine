import subprocess, psutil, time


# 创建一个子进程，例如运行一个Python程序
subprocess = subprocess.Popen(['python', 'CMI.py'])

# 循环监控子进程的CPU和内存使用情况
while subprocess.poll() is None:
    # 获取子进程信息
    process = psutil.Process(subprocess.pid)

    # 获取CPU使用率
    cpu_percent = process.cpu_percent(interval=1) / psutil.cpu_count()  # 单核CPU使用率

    # 获取内存使用情况
    memory_info = process.memory_info()
    rss = memory_info.rss / 1024 / 1024  # 常驻集大小，单位MB

    print(f'cpu count: {psutil.cpu_count()}, CPU usage: {cpu_percent}%')
    print(f'Memory usage: {rss} MB')

    time.sleep(0.002)  # 每秒检查一次

# 等待子进程结束
subprocess.wait()