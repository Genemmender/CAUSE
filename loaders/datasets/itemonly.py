from loaders.datasets.backbone import BackboneDataset
from loaders.env import Env
import torch


class ItemOnlyDataset(BackboneDataset):
    def build_inputs(
            self,
            group_length,  # 历史 group 数量（即 group_based 序列的长度）
            user_id: int,  # 当前样本的用户 id
            item_seq: list,  # item 序列（已 token 化）
            action_label_seq: list,  # 每个 item 对应的行为（如点击、点赞等）
            eval_seq: list,  # 每个位置是否参与评估
            feature_seq: dict,  # item 特征序列
            seen_mask: torch.Tensor,  # shape = [max_eval_nums, num_items]，记录 item 出现信息
    ):
        # 初始化输入序列：第一个位置为 user_id
        input_ids = []
        type_ids = []
        action_labels = []
        item_labels = []
        if self.use_separators:
            input_ids.append(self.SEG_USER_ID_INDEX)
            type_ids.append(self.sep_type)
            action_labels.append(Env.null)
            item_labels.append(Env.null)

        input_ids.append(user_id)
        type_ids.append(self.user_type)
        action_labels.append(Env.null)
        item_labels.append(Env.null)

        if self.use_separators:
            input_ids.append(self.SEG_HISTORY_INDEX)
            type_ids.append(self.sep_type)
            action_labels.append(Env.null)
            item_labels.append(Env.null)

        # 下一步 item 的预测目标（类似于 item_seq[1:]），末尾补 null 保持长度一致
        item_label_seq = item_seq[1:] + [Env.null]

        # 初始化每个特征列的输入序列（全是 padding）
        feature_ids = self.init_feature_ids(feature_seq)

        # 用 null 占位填充 group 部分
        for _ in range(group_length):
            input_ids.append(Env.null)
            type_ids.append(Env.null)
            action_labels.append(Env.null)
            item_labels.append(Env.null)
            self.fill_feature_ids(feature_ids, feature_seq)

        if self.use_separators:
            input_ids.append(self.SEG_ITEMS_INDEX)
            type_ids.append(self.sep_type)
            action_labels.append(Env.null)
            item_labels.append(Env.null)

        # 如果序列超长，则截断 item、label、action、eval 信息
        remaining_length = self.max_seq_item_num - group_length
        item_seq = item_seq[-remaining_length:]
        item_label_seq = item_label_seq[-remaining_length:]
        action_label_seq = action_label_seq[-remaining_length:]
        eval_seq = eval_seq[-remaining_length:]

        for feature in feature_seq:
            feature_seq[feature] = feature_seq[feature][-remaining_length:]

        seen_mask_index = 0  # 表示第几个用于评估的 item（用于索引 seen_mask）

        # 遍历每个 item，逐步填入输入序列
        for index, (item, action_label, item_label, for_eval) in enumerate(zip(
                item_seq, action_label_seq, item_label_seq, eval_seq)):
            input_ids.append(item)
            type_ids.append(self.item_type)

            # 仅在该 item 被用于评估时填充标签
            action_labels.append(action_label if for_eval else Env.null)
            current_item_label = item_label if for_eval else Env.null
            item_labels.append(current_item_label)

            # 构建 seen_mask：记录当前被评估的 item 之前出现过哪些 item
            if current_item_label != Env.null and self.global_validation:
                seq = torch.tensor(item_seq[:index], dtype=torch.long)
                seen_mask[seen_mask_index][seq] = True
                seen_mask_index += 1

            # 填充每个特征列的值
            for feature in feature_seq:
                feature_ids[feature].append(feature_seq[feature][index])
            # self.fill_feature_ids(feature_ids, feature_seq)

        # 返回所有构建好的输入张量
        return input_ids, type_ids, action_labels, item_labels, feature_ids, seen_mask_index
