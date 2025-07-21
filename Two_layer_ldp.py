import random
import numpy as np


def two_layer_dp(entity_type, entity_value, type_domain, value_domain_dict, epsilon1, epsilon2):
    """
    Two-layer DP mechanism for entity obfuscation.

    Args:
        entity_type (str): 原始实体类别，例如 "PERSON"。
        entity_value (str): 实体值，例如 "Alice"。
        type_domain (List[str]): 实体类别全集，例如 ["PERSON", "EMAIL_ADDRESS", "LOCATION", ...]。
        value_domain_dict (Dict[str, List[str]]): 每个实体类别对应的值域，例如 {"PERSON": ["Alice", "Bob", ...]}。
        epsilon1 (float): 第一层 DP（Type）的隐私预算。
        epsilon2 (float): 第二层 DP（Value）的隐私预算。

    Returns:
        str: obfuscated value，例如 "[PERSON]_Alice" 或 "[LOCATION]_Paris"
    """
    # --- 第一层: Entity Type 扰动 ---
    K1 = len(type_domain)
    p1 = np.exp(epsilon1) / (np.exp(epsilon1) + K1 - 1)
    if random.random() < p1:
        noisy_type = entity_type
    else:
        noisy_type = random.choice([t for t in type_domain if t != entity_type])

    # --- 第二层: Entity Value 扰动 ---
    value_domain = value_domain_dict.get(noisy_type, [])
    if entity_value in value_domain:
        true_value = entity_value
    else:
        # fallback: 如果 value 不在 domain 中，直接随机采样
        true_value = random.choice(value_domain) if value_domain else "UNKNOWN"

    K2 = len(value_domain)
    if K2 > 1:
        p2 = np.exp(epsilon2) / (np.exp(epsilon2) + K2 - 1)
        if random.random() < p2:
            noisy_value = true_value
        else:
            noisy_value = random.choice([v for v in value_domain if v != true_value])
    else:
        noisy_value = true_value

    # --- 生成 obfuscated 结果 ---
    return f"[{noisy_type}]_{noisy_value}"


type_domain = ["PERSON", "EMAIL_ADDRESS", "LOCATION", "PHONE_NUMBER"]
value_domain_dict = {
    "PERSON": ["Alice", "Bob", "Charlie",'Mike'],
    "LOCATION": ["Paris", "Tokyo", "New York"],
    "EMAIL_ADDRESS": ["user@example.com", "test@test.com"],
    "PHONE_NUMBER": ["123-456-7890", "555-555-5555"]
}
# 调用例子
# noisy_result = two_layer_dp(
#     entity_type="PERSON",
#     entity_value="Alice",
#     type_domain=type_domain,
#     value_domain_dict=value_domain_dict,
#     epsilon1=10,  # 第一层隐私预算
#     epsilon2=0.2   # 第二层隐私预算
# )
# print("DP Obfuscated Result:", noisy_result)
#
