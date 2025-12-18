from unitok import Symbol


class Symbols:
    """
    Symbols 类定义了项目中统一使用的 Symbol 实例（类似于枚举），
    主要用于标识训练/验证/测试状态、字段名、运行阶段、观察器操作等。
    通过 Symbol 保证全局一致性，避免硬编码字符串带来的错误。
    """

    # 模式相关
    train = Symbol('train')         # 训练模式
    val = Symbol('validate')        # 验证模式
    test = Symbol('test')           # 测试模式

    # 训练过程控制
    best = Symbol('best')           # 当前最优模型
    skip = Symbol('skip')           # 跳过当前 step/epoch
    stop = Symbol('stop')           # 停止训练

    # 数据类型标识
    user = Symbol('user')           # 用户
    item = Symbol('item')           # 物品

    # 模型输入输出字段
    input_ids = Symbol('input_ids')              # 输入 token ID
    type_ids = Symbol('type_ids')                # 类型 ID（对应 user/item/action 等）
    seg_ids = Symbol('segment_ids')              # 分段 ID
    feature_ids = Symbol('feature_ids')          # 特征 ID 字典
    action_labels = Symbol('action_labels')      # 行为标签
    item_labels = Symbol('item_labels')          # 物品标签（如多任务目标）
    attention_mask = Symbol('attention_mask')    # Attention mask
    seq_len = Symbol('seq_len')                  # 序列长度
    user_id = Symbol('user_id')                  # 用户 ID（用于验证/测试）
    index = Symbol('index')                      # 数据索引编号
    seen_mask = Symbol('seen_mask')              # 物品是否出现在用户历史中
    seen_mask_count = Symbol('seen_mask_count')  # 总共被评估的候选 item 数量

    # 群体表示相关字段
    group_length = Symbol('group_length')        # 每个用户对应的 group 数量
    group_index = Symbol('group_index')          # 群体 ID
    group_seq = Symbol('group_seq')              # 群体序列（item 序列）

    # 训练过程指标
    interval = Symbol('interval')                # 打印/评估间隔
    epoch = Symbol('epoch')                      # 当前 epoch 数

    # 观察器相关标识（如记录、存储）
    observer_push = Symbol('push')               # 推入观察器
    observer_pop = Symbol('pop')                 # 弹出观察器
    observer_export = Symbol('export')           # 导出观察器内容
    observer_interact = Symbol('interact')       # 交互式观察器
    observer_list = Symbol('list')               # 查看当前列表
    observer_rename = Symbol('rename')           # 重命名观察器
    observer_delete = Symbol('delete')           # 删除观察器

    # 当前运行状态标志
    is_initializing = Symbol('is_initializing') # 初始化状态
    is_training = Symbol('is_training')         # 正在训练状态
    is_evaluating = Symbol('is_evaluating')     # 正在评估状态
