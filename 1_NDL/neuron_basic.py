
class DendriteNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.axon_signal = 0  # 记录来自轴突的信号

    def calculate_signal(self):
        # 递归计算左子树和右子树的信号
        left_signal = self.left.calculate_signal() if self.left else 0
        right_signal = self.right.calculate_signal() if self.right else 0
        
        # 结合左、右子树信号和来自轴突的信号
        total_signal = 0.2*left_signal + 0.3*right_signal + 0.5*self.axon_signal
        
        # 这里可以自定义信号聚合的逻辑，例如简单的阈值判断
        return 1 if total_signal > 0.4 else 0  # 如果总信号大于1则输出1，否则输出0

    def set_axon_signal(self, signal):
        # 设置来自轴突的信号
        self.axon_signal = signal

class Dendrite:
    def __init__(self):
        # 创建一个两层的二叉树
        self.root = DendriteNode()
        self.root.left = DendriteNode()
        self.root.right = DendriteNode()

class Neuron:
    def __init__(self):
        self.dendrite = Dendrite()
        self.axon = []  # 记录连接到其它神经元的树突

    def init_signal(self, signal_self, signal_left, signal_right):
        # 设置树突信号
        self.dendrite.root.axon_signal = signal_self
        self.dendrite.root.left.axon_signal = signal_left
        self.dendrite.root.right.axon_signal = signal_right


    def process_signals(self):
        # 计算胞体信号
        self.cell_body_signal = self.dendrite.root.calculate_signal()

    def connect_to(self, other_neuron, target_dendrite_node):
        # 连接到其它神经元的指定树突节点
        self.axon.append((other_neuron, target_dendrite_node))

    def transmit_signal(self):
        # 将信号传递给连接的神经元的树突
        for target_neuron, target_node in self.axon:
            target_node.set_axon_signal(self.cell_body_signal)  # 传递相同的信号

if __name__ == "__main__":
    neuron1 = Neuron()
    neuron2 = Neuron()

    # neuron1接收信号
    neuron1.init_signal(1, 1, 0)  # 左子树信号为1，右子树信号为0
    neuron1.process_signals()  # 处理信号

    # 连接到neuron2的左子树
    neuron1.connect_to(neuron2, neuron2.dendrite.root.left)
    neuron1.connect_to(neuron2, neuron2.dendrite.root)

    # 传递信号
    neuron1.transmit_signal()

    # 计算neuron2的信号
    neuron2.process_signals()

    # 检查 neuron 的胞体信号
    print(f"Neuron1 cell body signal: {neuron1.cell_body_signal}")
    print(f"Neuron2 cell body signal: {neuron2.cell_body_signal}")