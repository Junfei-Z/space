import Two_layer_ldp
from edge_utils import EdgeProcessor

def main():
    processor = EdgeProcessor()

    # 输入 prompt
    prompt = "Hi, my name is Alice. I live in Paris and my email is alice@example.com."

    # 1️⃣ NER 识别
    entities = processor.run_ner(prompt)
    print(f"NER Entities: {entities}")

    # 2️⃣ 风险分数判断
    risk = processor.compute_risk_score(entities)
    print(f"Risk Score: {risk:.2f}")

    if risk < processor.risk_threshold:
        print("Low risk. No encryption needed.")
        protected_prompt = prompt
    else:
        print("Sensitive. Applying Two-Layer DP...")
        protected_prompt = Two_layer_ldp.two_layer_dp(prompt, entities)
    print(f"Protected Prompt: {protected_prompt}")

    # 3️⃣ 提取 embedding
    embedding = processor.extract_embedding(protected_prompt)
    print("Embedding extracted. Sending to cloud...")

    # 4️⃣ 发送到云端并接收 sketch
    sketch = processor.send_to_cloud(embedding)
    print(f"Received Sketch: {sketch}")

    # 5️⃣ 客户端补齐（简单拼接示意）
    final_response = sketch.replace("[MASK]", "Alice")  # 可结合之前保留的 entity 做精确补齐
    print(f"Final Response: {final_response}")

if __name__ == "__main__":
    main()
