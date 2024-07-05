from typing import List
import dataclasses
from petals_tensor.client.routing.sequence_manager import MissingBlocksError
from petals_tensor.data_structures import RemoteSpanInfo
from petals_tensor.health.state_updater import get_peer_ids_list
from petals_tensor.server.server import Server
from petals_tensor.substrate.chain_functions import propose_model_peer_dishonest, vote_model_peer_dishonest
from petals_tensor.utils.auto_config import AutoDistributedModelForCausalLM
from petals_tensor.substrate import config as substrate_config
import numpy as np
import torch
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)

# TODO: make substrate_config a class
# from hypertensor import HypertensorClient

class InferenceValidator:
    def __init__(self, server: Server, model_name, peers, node_url):
        self.server = server
        self.model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
        self.client = substrate_config.SubstrateConfig
        self.peer_ids = None
        self.validated_peer_ids = None
        self.epoch = 0

    def run(self):
        while True:
            """Check current epoch, ensure epoch not already validated"""
            epoch = self._get_epoch()

            if epoch == self.epoch:
                # set to wait remaining time period
                continue
        
            self.epoch = epoch

            """Set as validator for full control over blocks"""
            self.server.validator = True

            """Get all peers and update list of peers for this epoch"""
            self.update_peers()
            try:
                while True:
                    """ """
                    peer_ids = self.select_peers()

                    # self.make_sequence_with_specific_peer(peer_ids, start_index: int, end_index: int)
            finally:
                """ """
                self.server.validator = False

    def select_peers(self):
        # Logic to select peers for validation
        return self.peers[:2]  # Example: select the first two peers

    def update_blocks(self, peer_blocks):
        # Logic to update validator's blocks to match peer's blocks
        self.blocks = peer_blocks

    def set_deterministic(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def validate_inference(self, input_data, expected_outputs):
        # Perform inference
        outputs = self.model.generate(
            input_data, 
            self.peer_ids, 
            max_new_tokens=5
        )

        # Check if the output matches the expected output
        return outputs == expected_outputs

    def initiate_dishonesty(self, peer_id):
        # Propose the peer as dishonest on the blockchain

        # Check if proposal already exists
        proposal_exists = True 
        if proposal_exists:
            tx_hash = vote_model_peer_dishonest(
                self.client.substrate_interface,
                self.client.keypair,
                model_id=0,  # Example: model id
                peer_id=peer_id,  # Example: peer id to vote as dishonest
            )
        else:
            tx_hash = propose_model_peer_dishonest(
                self.client.substrate_interface,
                self.client.keypair,
                model_id=0,  # Example: model id
                peer_id=peer_id,  # Example: peer id to vote as dishonest
            )

        # print(f"Proposed dishonest peer {peer_id} with transaction hash: {tx_hash}")

    def recalibrate_blocks(self):
        # Logic to recalibrate blocks to the most optimized state
        print("Recalibrating blocks to the most optimized state")

    def update_peers(self):
        self.peers = []
        peer_ids_list = get_peer_ids_list()
        for peer in peer_ids_list:
            self.peers.append(peer)

    # def make_sequence(self, peer_ids: List[str], start_index: int, end_index: int) -> List[RemoteSpanInfo]:
    #     client_server_rtts = self.ping_aggregator.to_dict()

    #     span_sequence = []
    #     current_index = start_index
    #     while current_index < end_index:
    #         candidate_spans = self.state.sequence_info.spans_containing_block[current_index]
    #         if not candidate_spans:
    #             raise MissingBlocksError(current_index)

    #         if any(span.peer_id in set(peer_ids) for span in candidate_spans):
    #             peer = None
    #             for span in candidate_spans:
    #                 for peer_id in peer_ids:
    #                     if span.peer_id == peer_id:
    #                         logger.info(f"Found PeerId in sequence {peer_id}")
    #                         peer = span
    #                         # Include peer in sequence and remove peer from peer_ids
    #                         peer_ids.remove(peer_id)
    #                         break

    #             chosen_span = peer
    #         else:
    #             # We choose longer servers to minimize the number of hops but leave some randomization
    #             # to distribute the load. We also exclude servers known to be unreachable.
    #             eps = 1e-6
    #             span_weights = np.array(
    #                 [span.length if client_server_rtts.get(span.peer_id) != np.inf else eps for span in candidate_spans],
    #                 dtype=np.float64,
    #             )
    #             chosen_span = np.random.choice(candidate_spans, p=span_weights / span_weights.sum())

    #         assert chosen_span.start <= current_index < chosen_span.end
    #         span_sequence.append(dataclasses.replace(chosen_span, start=current_index))
    #         current_index = chosen_span.end
    #     return span_sequence
    
    def _get_epoch(self):
        return 1
