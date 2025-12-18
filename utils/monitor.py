from loaders.symbols import Symbols as Syb  # 定义控制信号的常量类，比如 Syb.stop / Syb.best / Syb.skip


class Monitor:
    def __init__(self, minimize, patience=2, epoch=None):
        """
        初始化监控器，用于训练过程中的 Early Stopping 控制逻辑。

        参数：
            minimize (bool): 指标是否是“越小越好”，例如 LogLoss 需要 minimize=True。
            patience (int): 若性能连续多少步未提升则停止（可选）。
            epoch (int): 训练达到指定轮数后强制停止（可选）。

        注意：
            patience 和 epoch 至少要指定一个，否则抛出异常。
        """
        self.patience = patience
        self.epoch = epoch

        if not patience and not epoch:
            raise ValueError("Either patience or epoch must be specified.")

        self._best_value = None     # 当前最优值（根据指标）
        self._minimize = minimize  # 是否是最小化问题
        self._best_index = 0       # 最佳值出现的步数（或 epoch）
        self._current_index = -1   # 当前步数索引（每次 push +1）

    def push(self, value, epoch):
        """
        推入一个新的评估值，并返回当前监控状态。

        参数：
            value: 当前 epoch 的指标值（如 LogLoss、AUC）
            epoch: 当前轮次

        返回：
            Syb.best: 当前是最优表现，应该保存模型
            Syb.skip: 无需保存模型
            Syb.stop: 满足 early stopping 条件，应该终止训练
        """
        self._current_index += 1

        # 达到最大 epoch 限制，强制停止
        if self.epoch and epoch >= self.epoch:
            return Syb.stop

        # 不设置 early stopping，始终保存最优
        if not self.patience:
            return Syb.best

        # 第一次记录，初始化最优值
        if self._best_value is None:
            self._best_value = value
            return Syb.best

        # 判断当前是否优于历史最优
        improved = self._minimize ^ (value > self._best_value)
        if improved:
            self._best_value = value
            self._best_index = self._current_index
            return Syb.best

        # 若已经 patience 轮没有提升，则停止训练
        if self._current_index - self._best_index >= self.patience:
            return Syb.stop

        # 否则跳过当前模型保存
        return Syb.skip
