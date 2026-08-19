# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 19:07:46 2026

For a dataset that has values only, such as one created with convert_real_games.py,
Compute MCTS per game from a fresh tree and save the probability buffer to the
dataset. Save it as a new file with a new name.

@author: jledragon
"""

from pathlib import Path
import argparse
import torch
import torch.multiprocessing as mp
import chess_cpp
from agents import A2CMoveAgent, A2CGameMemory, MCTSGraph
from neural_networks import A2CChessNetwork

NUM_GAMES_PER_PROCESS = 10  # The workload to be given to one thread. Should be small and manageable.

NUM_PROCESSES = 5


def get_args():
    parser = argparse.ArgumentParser(description='Parameters for adding the probability buffer to value datasets.')
    parser.add_argument(
        'file_name',
        type=str,
        help="The name of the file that contains the data."
    )
    parser.add_argument(
        '-mcts_depth',
        type=int,
        default=800,
        help='Number of simulations per run of MCTS.'
    )
    parser.add_argument(
        '--disallow_noise',
        action='store_true',
        help='Whether to prevent Dirichlet noise being added when running MCTS simulations.',
    )
    args = parser.parse_args()
    return args


def multi_threaded_a2c_process(proc_num, state_batch, num_times_seen, disallow_noise, depth):
    boards = chess_cpp.BatchedBoard(True, 1, 0)
    model = A2CChessNetwork()  # Later - add in a model state, if wanted
    prob_dist = []
    for i in range(state_batch.shape[0]):
        # Load the data, generate the graph
        board_i = state_batch[i:i+1,:,:,:].cuda()
        nts = num_times_seen[i:i+1].item()
        our_ai_agent = A2CMoveAgent(boards, model, board_i, {}, False)
        our_ai_agent.prepare_for_evaluation()
        mcts = MCTSGraph(our_ai_agent, boards)
        mcts.depth = depth + (nts - 1)  # Give a little more importance for states that appear a lot.
        move_layer = chess_cpp.get_moves_for_player(board_i)
        mcts.init_top_node_if_empty_graph(board_i, move_layer)
        mcts.generate_graph(not disallow_noise)

        # Get the probability distribution
        child_n_values = [child.N for child in mcts.top_node.children]
        temperature_values = [(n / mcts.top_node.N) ** mcts.top_node.temperature for n in child_n_values]
        sum_temp = sum(temperature_values)
        norm_temp = torch.Tensor([t / sum_temp for t in temperature_values]) * nts
        prob_dist.append(norm_temp)
    return proc_num, prob_dist


if __name__ == '__main__':
    args = get_args()
    assert torch.cuda.is_available(), "CUDA is not enabled. Please fix this before running this script."
    torch._dynamo.config.cache_size_limit = 64
    memory = A2CGameMemory(256, val_only_mode=True)
    memory.load_data(args.file_name)
    mcts_tmp = Path.cwd() / 'datasets' / f'{args.file_name}_a2c_prob_tmp.pt'
    if mcts_tmp.exists():
        data_map = torch.load(mcts_tmp)
        a2c_prob_distribution = data_map["a2c_prob_distribution"]
        start = len(a2c_prob_distribution)
    else:
        data_map = {}
        a2c_prob_distribution = []
        start = 0
    print(f"Beginning from {start}. Trying to reach {memory.state_buffer.shape[0]}.")

    agent_pool = mp.Pool(processes=NUM_PROCESSES)
    while len(a2c_prob_distribution) < memory.state_buffer.shape[0]:
        print(f"Beginning another batch of {NUM_GAMES_PER_PROCESS * NUM_PROCESSES} states from {start}.")
        a2c_prob_distribution += [None] * NUM_GAMES_PER_PROCESS * NUM_PROCESSES
        process_inputs = [
            (
                i,
                memory.state_buffer[start+i*NUM_GAMES_PER_PROCESS : start+(i+1)*NUM_GAMES_PER_PROCESS,:,:],
                memory.num_times_seen[start+i*NUM_GAMES_PER_PROCESS : start+(i+1)*NUM_GAMES_PER_PROCESS],
                args.disallow_noise,
                args.mcts_depth,
            ) for i in range(NUM_PROCESSES)
        ]
        process_values = agent_pool.starmap(multi_threaded_a2c_process, process_inputs)
        for (proc_num, prob_dist) in process_values:
            begin = start + proc_num * NUM_GAMES_PER_PROCESS
            for pi, a2c_prob_dist in enumerate(prob_dist):
                a2c_prob_distribution[begin + pi] = a2c_prob_dist
        start += NUM_GAMES_PER_PROCESS * NUM_PROCESSES
        data_map["a2c_prob_distribution"] = a2c_prob_distribution
        torch.save(data_map, mcts_tmp)  # Do this regularly.
    agent_pool.close()

    a2c_prob_distribution = a2c_prob_distribution[0:memory.state_buffer.shape[0]]
    memory.mcts_prob_buffer = a2c_prob_distribution
    memory.save_data()
