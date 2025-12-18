import torch

from loaders.datasets.backbone import BackboneDataset
from loaders.env import Env


class FlattenDataset(BackboneDataset):
    def get_max_length(self):
        # 输入序列最大长度：每个 item 会变成 item + label 两个 token，外加一个 user_id token
        return self.max_seq_item_num * 2 + 1 + (3 if self.use_separators else 0)

    def build_inputs(
            self,
            group_length,  # 表示历史行为中 group 的数量，用于跳过群体信息部分
            user_id: int,
            item_seq: list,  # item 序列
            action_label_seq: list,  # 对应每个 item 的点击/行为标签
            eval_seq: list,  # 表示哪些位置参与评估（例如 AUC、NDCG）
            feature_seq: dict,  # item 的额外特征（例如 tag, music_type 等）
            seen_mask: torch.Tensor,  # shape: [max_eval_nums, num_items]，用于记录哪些 item 在历史中出现过
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

        # item_labels 预测的是下一步 item（对应 item_seq[1:]），最后补上 null
        item_label_seq = item_seq[1:] + [Env.null]

        # 初始化特征输入结构（带空 padding）
        feature_ids = self.init_feature_ids(feature_seq)

        # group_length 部分全部填充 null（用于占位）
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

        # 根据剩余序列长度，进行截断
        remaining_length = self.max_seq_item_num - group_length
        item_seq = item_seq[-remaining_length:]
        item_label_seq = item_label_seq[-remaining_length:]
        action_label_seq = action_label_seq[-remaining_length:]
        eval_seq = eval_seq[-remaining_length:]
        for feature in feature_seq:
            feature_seq[feature] = feature_seq[feature][-remaining_length:]

        seen_mask_index = 0  # 用于记录第几个 item 被用于评估（有正样本）

        # 遍历 item 序列，展开成 [item, action] 格式，并生成标签
        for index, (item, action_label, item_label, for_eval) in enumerate(zip(
                item_seq, action_label_seq, item_label_seq, eval_seq)):

            # 输入部分：每个 item 拆成 item 和 action 两个 token
            input_ids += [item, action_label]
            type_ids += [self.item_type, self.label_type]

            # 如果该 item 参与评估（for_eval=True），则保留标签，否则填充 null
            action_labels += [action_label if for_eval else Env.null, Env.null]

            # item_labels 仅在当前 item 参与评估时才写入当前 label，用于 item-level 评估任务
            current_item_label = item_label if for_eval else Env.null
            item_labels += [Env.null, current_item_label]

            # seen_mask：记录当前评估 item 之前出现过哪些 item（用于 AUC、NDCG 等历史曝光约束）
            if current_item_label != Env.null and self.global_validation:
                seq = torch.tensor(item_seq[:index], dtype=torch.long)
                seen_mask[seen_mask_index][seq] = True
                seen_mask_index += 1

            # 填充特征（每个 item 一个）
            for feature in feature_seq:
                feature_ids[feature].append(feature_seq[feature][index])
            # 填充 action 部分（等长）
            self.fill_feature_ids(feature_ids, feature_seq)

        return input_ids, type_ids, action_labels, item_labels, feature_ids, seen_mask_index
