from loaders.datasets.backbone import BackboneDataset
from loaders.env import Env
import torch


class StackedDataset(BackboneDataset):
    def build_inputs(
            self,
            group_length,  # 历史 group 数量
            user_id: int,  # 当前用户 id
            item_seq: list,  # item 序列
            action_label_seq: list,  # item 对应的行为标签
            eval_seq: list,  # 是否用于评估
            feature_seq: dict,  # item 特征（dict of list）
            seen_mask: torch.Tensor,  # 每个评估点前出现过的 item
    ):
        # 初始化输入序列：第一个位置为 user_id
        input_ids = []
        type_ids = []
        action_labels = []
        item_labels = []
        if self.use_separators:
            input_ids.append([self.SEG_USER_ID_INDEX, 0])
            type_ids.append([self.sep_type, Env.null])
            action_labels.append(Env.null)
            item_labels.append(Env.null)

        input_ids.append([user_id, 0])
        type_ids.append([self.user_type, Env.null])
        action_labels.append(Env.null)
        item_labels.append(Env.null)

        if self.use_separators:
            input_ids.append([self.SEG_HISTORY_INDEX, 0])
            type_ids.append([self.sep_type, Env.null])
            action_labels.append(Env.null)
            item_labels.append(Env.null)

        # 初始化输入编码：首位为 [user_id, 0]，类型为 [user_type, null]
        # 使用二元嵌套结构：input_ids[i] = [item, action]，type_ids[i] = [type, type]
        # input_ids = [[user_id, 0]]
        # type_ids = [[self.user_type, Env.null]]

        # # 初始化标签，首位无标签
        # action_labels = [Env.null]
        # item_labels = [Env.null]

        # 目标 item label（预测 item_seq 的下一个）
        item_label_seq = item_seq[1:] + [Env.null]

        # 初始化特征字段（全部填 null）
        feature_ids = self.init_feature_ids(feature_seq)

        # Group 部分用 null 占位填充（group_length 行）
        for _ in range(group_length):
            input_ids.append([Env.null, Env.null])
            type_ids.append([Env.null, Env.null])
            action_labels.append(Env.null)
            item_labels.append(Env.null)
            self.fill_feature_ids(feature_ids, feature_seq)

        if self.use_separators:
            input_ids.append([self.SEG_ITEMS_INDEX, 0])
            type_ids.append([self.sep_type, Env.null])
            action_labels.append(Env.null)
            item_labels.append(Env.null)

        # 超长序列截断
        remaining_length = self.max_seq_item_num - group_length
        item_seq = item_seq[-remaining_length:]
        item_label_seq = item_label_seq[-remaining_length:]
        action_label_seq = action_label_seq[-remaining_length:]
        eval_seq = eval_seq[-remaining_length:]
        for feature in feature_seq:
            feature_seq[feature] = feature_seq[feature][-remaining_length:]

        seen_mask_index = 0  # 表示第几个用于评估的 item

        # 遍历每个 item，构建 input_ids、标签等
        for index, (item, action_label, item_label, for_eval) in enumerate(
                zip(item_seq, action_label_seq, item_label_seq, eval_seq)):
            # 将 item 和 action_label 同时编码为一对
            # 若该位置为评估点，则 action_label 填 Env.null，避免信息泄露
            input_ids.append([item, action_label if not for_eval else Env.null])
            type_ids.append([self.item_type, self.label_type if not for_eval else Env.null])

            # 如果该位置用于评估，则 action_label / item_label 是监督信号
            action_labels.append(action_label if for_eval else Env.null)

            current_item_label = item_label if for_eval else Env.null
            item_labels.append(current_item_label)

            # 如果当前为有效评估位置，构建 appear_flag
            if current_item_label != Env.null and self.global_validation:
                seq = torch.tensor(item_seq[:index], dtype=torch.long)  # 当前样本的历史 item
                seen_mask[seen_mask_index][seq] = True
                seen_mask_index += 1

            # 填充所有特征字段
            for feature in feature_seq:
                feature_ids[feature].append(feature_seq[feature][index])

        # 返回最终构建好的所有输入
        return input_ids, type_ids, action_labels, item_labels, feature_ids, seen_mask_index
