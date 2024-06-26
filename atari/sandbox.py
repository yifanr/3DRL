import gzip
import cv2
import numpy as np
import atari_py

STORE_FILENAME_PREFIX = '$store$_'

ELEMS = ['observation', 'action', 'reward', 'terminal']

if __name__ == '__main__':

    data = {}

    data_dir = '../dqn_replay/Breakout/1/replay_logs/'
    suffix = 0
    for elem in ELEMS:
        filename = f'{data_dir}{STORE_FILENAME_PREFIX}{elem}_ckpt.{suffix}.gz'
        with open(filename, 'rb') as f:
            with gzip.GzipFile(fileobj=f) as infile:
                data[elem] = np.load(infile)
                print(data[elem].shape)
    action_weight = 10

    print(data['observation'][100][10:20])
    cv2.imwrite('dummy.png', data['observation'][100]) 
    


    # ale = atari_py.ALEInterface()
    # ale.setInt('random_seed', 123)
    # ale.setInt('max_num_frames_per_episode', 108e3)
    # ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
    # ale.setInt('frame_skip', 0)
    # ale.setBool('color_averaging', False)
    # ale.loadROM(atari_py.get_game_path('breakout')) 
    # actions = ale.getMinimalActionSet()
    # print(actions)
    # state = cv2.resize(ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    # print(state.shape)
    # cv2.imwrite('state.png', state) 

    # for obs in data['observation']:
    #     # cv2.imshow('obs', obs)
    #     cv2.imwrite('dummy.png', obs) 