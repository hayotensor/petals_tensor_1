"""
Return peer data stored in the substrate config

Note: In dev mode the blockchain must be running

python -m petals_tensor.cli.run_test_inference
"""
import argparse
import logging
from petals_tensor.client import InferenceSession
from petals_tensor.utils.auto_config import AutoDistributedModelForCausalLMValidator
from transformers import AutoTokenizer
from petals_tensor import AutoDistributedModelForCausalLM
# from petals_tensor.consensus import InferenceSessionValidator


logger = logging.getLogger(__name__)

def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      "--peer",
      nargs="*",
      help="Multiaddrs of the peers that will welcome you into the existing DHT. "
      "Example: /ip4/203.0.113.1/tcp/31337/p2p/XXXX /ip4/203.0.113.2/tcp/7777/p2p/YYYY",
  )
  args = parser.parse_args()

  # Choose any model available at https://health.petals.dev
  model_name = "petals-team/StableBeluga2"  # This one is fine-tuned Llama 2 (70B)
  
  print(model_name)
  # Connect to a distributed network hosting model layers
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  print("tokenizer", tokenizer)

  # model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
  # print("model", model)

  model = AutoDistributedModelForCausalLMValidator.from_pretrained(model_name)
  print("model", model)

  # Run the model as if it were on your computer
  inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"]
  print("inputs", inputs)

  # outputs = model.generate_with_specific_peer(inputs, max_new_tokens=5)
  # outputs = model.generate_with_specific_peer("12D3KooWKkY4mz7riahcMo7NUnfhtFrAcexfLGZigFRS3MtaKXKo", inputs, max_new_tokens=5)
  # outputs = model.generate(inputs, peer_id="12D3KooWMGqMt6GEeqwgBhkrHPH5sT7zrBmMjHMMoyGz1b5ahE3f", max_new_tokens=5)
  # print("outputs", outputs)

  outputs = model.generate(inputs, peer_ids=["12D3KooWCWrjfLzYL4zJevm35MzexAwjngE1WNPYeiHrsLTGEkoy"], max_new_tokens=5)
  print("outputs", outputs)

  print(tokenizer.decode(outputs[0]))  # A cat sat on a mat...




if __name__ == "__main__":
    main()