import matplotlib.pyplot as plt

def plot_loss(log_file1,log_file2,log_file3,log_file4):
    # 读取日志文件
    iterations1 = []
    losses1 = []
    bcs1 = []
    res1 = []
    times1= []
    l2s1 = []

    iterations2 = []
    losses2 = []
    bcs2 = []
    res2 = []
    times2= []
    l2s2 = []

    iterations3 = []
    losses3 = []
    bcs3 = []
    res3 = []
    times3= []
    l2s3= []

    iterations4 = []
    losses4 = []
    bcs4 = []
    res4 = []
    times4= []
    l2s4= []

    with open(log_file1, "r") as f:
        lines = f.readlines()[1:]  # 跳过表头
        for line in lines:
            iteration, loss, bc, res, time ,l2= line.strip().split(", ")
            iterations1.append(int(iteration))
            losses1.append(float(loss))
            bcs1.append(float(bc))
            res1.append(float(res))
            times1.append(float(time))
            l2s1.append(float(l2))
    with open(log_file2, "r") as f:
        lines = f.readlines()[1:]  # 跳过表头
        for line in lines:
            iteration, loss, bc, res, time ,l2= line.strip().split(", ")
            iterations2.append(int(iteration))
            losses2.append(float(loss))
            bcs2.append(float(bc))
            res2.append(float(res))
            times2.append(float(time))
            l2s2.append(float(l2))
    with open(log_file3, "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            iteration, loss, bc, res, time ,l2= line.strip().split(", ")
            iterations3.append(int(iteration))
            losses3.append(float(loss))
            bcs3.append(float(bc))
            res3.append(float(res))
            times3.append(float(time))
            l2s3.append(float(l2))

    with open(log_file4, "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            iteration, loss, bc, res, time ,l2= line.strip().split(", ")
            iterations4.append(int(iteration))
            losses4.append(float(loss))
            bcs4.append(float(bc))
            res4.append(float(res))
            times4.append(float(time))
            l2s4.append(float(l2))

    # 绘制loss曲线
    plt.figure(figsize=(6, 6))
    # plt.plot(times, losses, label="Loss")
    # plt.plot(times1, losses1, label="Loss from log_file1", color='blue')
    # plt.plot(times2, losses2, label="Loss from log_file2", color='red')
    # plt.plot(iterations1,losses1, label="Loss from log_file1", color='blue')
    # plt.plot(iterations2,losses2,  label="Loss from log_file2", color='red')
    plt.plot(times1, l2s1, label="PINN", color='blue')

    plt.plot(times3, l2s3, label="LAAF-PINN", color='green')
    plt.plot(times4, l2s4, label="GAAF-PINN", color='orange')
    plt.plot(times2, l2s2, label="frecPINN", color='red')
    # plt.plot(times1, losses1, label="Loss from log_file1", color='blue')
    # plt.plot(times2, losses2, label="Loss from log_file2", color='red')
    # plt.plot(times3, losses3, label="Loss from log_file3", color='green')
    # plt.plot(times4, losses4, label="Loss from log_file4", color='orange')

    # plt.plot(iterations1, l2s1, label="PINN", color='blue')
    # # plt.plot(iterations2, l2s2, label="frecPINN", color='red')
    # plt.plot(iterations3, l2s3, label="LAAF-PINN", color='green')
    # plt.plot(iterations4, l2s4, label="GAAF-PINN", color='orange')
    # plt.plot(iterations2, l2s2, label="frecPINN", color='red')
    # plt.plot(times1, bcs1, label="bcs", color='blue')
    # plt.plot(times1, res1, label="res", color='red')
    # plt.plot(times1, losses1, label="loss", color='orange')
    # plt.plot(times1, res1, label="Loss from log_file1", color='blue')
    # plt.plot(times2, res2, label="Loss from log_file2", color='red')

    # plt.plot(times1, bcs1, label="Loss from log_file1", color='blue')
    # plt.plot(times2, bcs2, label="Loss from log_file2", color='red')

    plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('Time (seconds)')
    # plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Relative L2 Error')
    plt.grid(True)
    plt.legend()
    plt.show()


plot_loss(log_file1='loss_fuck_p.txt',log_file2='loss_fuck_frp.txt',log_file3='loss_laafburgrl2(x0000).txt',log_file4='loss_gf.txt')
