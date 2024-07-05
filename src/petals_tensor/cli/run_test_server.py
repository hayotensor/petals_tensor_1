"""
Return peer data stored in the substrate config

Note: In dev mode the blockchain must be running

python -m petals_tensor.cli.run_test_server
"""
import argparse
import logging
from petals_tensor.client import InferenceSession
from petals_tensor.constants import PUBLIC_INITIAL_PEERS
from petals_tensor.data_structures import UID_DELIMITER
from petals_tensor.server import block_selection
from petals_tensor.server.server import Server
from petals_tensor.utils.auto_config import AutoDistributedConfig
from petals_tensor.utils.dht import get_remote_module_infos
from petals_tensor.utils.disk_cache import DEFAULT_CACHE_DIR
from transformers import AutoTokenizer
from petals_tensor import AutoDistributedModelForCausalLM
# from petals_tensor.consensus import InferenceSessionValidator
from hivemind import DHT, PeerID

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

  block_config = AutoDistributedConfig.from_pretrained(
    model_name_or_path=model_name,
    use_auth_token=None,
    revision=None,
  )

  dht = DHT(
    initial_peers=PUBLIC_INITIAL_PEERS,
    start=True,
    num_workers=block_config.num_hidden_layers,
    use_relay=True,
    use_auto_relay=True,
    client_mode=True,
  )

  dht_prefix="StableBeluga2-hf"
  # module_uids = [f"{dht_prefix}{UID_DELIMITER}{block_index}" for block_index in block_indices]

  cache_dir = DEFAULT_CACHE_DIR

  module_uids = [
    f"{dht_prefix}{UID_DELIMITER}{block_index}"
    for block_index in range(block_config.num_hidden_layers)
  ]

  peer_id = PeerID.from_base58("12D3KooWCFGUYe6QZKHQayQqWDGxRtjx1ouTmp2YwWSZZfWdUB9C")
  module_infos = get_remote_module_infos(dht, module_uids, latest=True)
  should_choose_other_blocks_ =  block_selection.should_choose_other_blocks(peer_id, module_infos, 0.75)
  print("should_choose_other_blocks_", should_choose_other_blocks_)

  # block_indices=range(14, 27), # this is where the blocks are updated

  server = Server(
    converted_model_name_or_path=model_name,
    initial_peers=PUBLIC_INITIAL_PEERS,
    dht_prefix=None,
    block_indices="14:27",
    throughput=0.75,
    num_handlers=8,
    min_batch_size=1,
    max_chunk_size_bytes=256 * 1024 * 1024,
    max_alloc_timeout=600,
    cache_dir=DEFAULT_CACHE_DIR,
    torch_dtype="auto",
    compression=120,
    request_timeout=3 * 60,
    session_timeout=30 * 60,
    step_timeout=5 * 60,
    prefetch_batches=1,
    sender_threads=1,
    balance_quality=0.75,
    mean_balance_check_period=60,
    skip_reachability_check=True,
    use_relay=True,
    use_auto_relay=True,
    adapters=(),
  )
  try:
    server.run()
  except Exception as e:
    logger.info(f"Error running {e}")

if __name__ == "__main__":
    main()