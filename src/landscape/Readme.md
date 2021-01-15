# Visualize Loss Landscape


在src目录下，运行 python landscape/vis_loss_landscape.py

在运行之前，修改vis_loss_landscape.py：
0. **注意**，我为了加速，在第217行加了个if step>10: break. 这句应该去掉
1. 修改第324行 base_dir路径，这个路径是存放ckpt的路径（我这里是把一个模型文件单独放在一个文件夹下，以为我还会生成针对该模型文件的direction文件和results文件）
2. 可以改vis_loss_landscape.py中的参数，比如skip_beta
3. parser中的x,y分别表示在两个方向上扰动的范围，默认是-1～1，我为了加速平均取5个点，这里最好是取51个点（nips论文中是51个点）

在第一次运行这个文件时，程序会计算在test数据集上的loss和acc，然后存到results.csv中，之后程序会自动找是否存在results.csv文件，存在的话就直接读取results.csv中的值，画图（因为画图的时候，可能需要调节等高线显示的采样点，所以可能需要多试几次选一个最好的）