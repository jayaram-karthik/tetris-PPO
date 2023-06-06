import numpy as np
import random
from copy import deepcopy
import cv2

def get_score(lines_cleared):
    if lines_cleared == 1:
        return 4
    elif lines_cleared == 2:
        return 10
    elif lines_cleared == 3:
        return 30
    elif lines_cleared == 4:
        return 120
    return 0

class TetrisEnv:
    actions = {
        0: 'shift_left', 
        1: 'shift_right',
        2: 'rotate_counterclockwise',
        3: 'rotate_clockwise',
        4: 'soft_drop',
        5: 'hard_drop',
        6: 'none'
    }
    
    pieces = {
        'I': [(0, 0), (0, -1), (0, -2), (0, -3)],
        'O': [(0, 0), (0, -1), (-1, 0), (-1, -1)],
        'S': [(0, 0), (1, 0), (-1, -1), (0, -1)],
        'L': [(0, 0), (1, 0), (0, -1), (0, -2)],
        'T': [(0, 0), (-1, 0), (1, 0), (0, -1)],
        'J': [(0, 0), (-1, 0), (0, -1), (0, -2)],
        'Z': [(0, 0), (-1, 0), (0, -1), (1, -1)],
    }
    
    WIDTH = 10
    HEIGHT = 20
    
    NUM_ACTIONS = 7
    
    
    
    def __init__(self):
        self.n_frames = 0
        self.score = 0
        
        self.curr_shape = None # anchor shape
        self.shape_in_play = None
        
        self.att_num = 1
        
        self.board = np.zeros((self.HEIGHT, self.WIDTH))
        
        
        return
    
    
    def _rotate_ccw_helper(self, piece):
        return [(-y, x) for x, y in piece]
    
    def _rotate_cw_helper(self, piece):
        return [(y, -x) for x, y in piece]
    
    def gen_new_shape(self):
        shape = random.choice(list(self.pieces.keys()))
        shape = self.pieces[shape]
        
        return shape
    
    def is_current_piece_dropped(self):
        return
    
    def is_colliding(self, shape_structure, shape_coords):
        for i, j in shape_structure:
            pixel_x, pixel_y = shape_coords[0] + i, shape_coords[0] + j
            
            if pixel_y < 0:
                continue
            elif pixel_x < 0 or pixel_x >= self.board.shape[0] or pixel_y >= self.board.shape[1] or self.board[pixel_x, pixel_y] != 0:
                return True
        
        return False
    
    def shift_left(self):
        if self.is_colliding(self.shape_in_play, self.curr_shape):
            return (self.shape_in_play, self.curr_shape)
        
        return (self.shape_in_play, (self.curr_shape[0] - 1, self.curr_shape[1]))
    
    def shift_right(self):
        if self.is_colliding(self.shape_in_play, self.curr_shape):
            return (self.shape_in_play, self.curr_shape)
        
        return (self.shape_in_play, (self.curr_shape[0] + 1, self.curr_shape[1]))
    
    def soft_drop(self):
        if self.is_colliding(self.shape_in_play, self.curr_shape):
            return (self.shape_in_play, self.curr_shape)
        
        return (self.shape_in_play, (self.curr_shape[0], self.curr_shape[1] + 1))
    
    def hard_drop(self):
        while True:
            _, new_coords = self.soft_drop()
            if new_coords == self.curr_shape:
                return self.shape_in_play, new_coords
            self.shape_in_play = new_coords
    
    def rotate_counterclockwise(self):
        new_shape = self._rotate_ccw_helper(self.shape_in_play)
        if self.is_colliding(new_shape, self.curr_shape):
            return (self.shape_in_play, self.curr_shape)
        
        return (new_shape, self.curr_shape)
    
    def rotate_clockwise(self):
        new_shape = self._rotate_cw_helper(self.shape_in_play)
        if self.is_colliding(new_shape, self.curr_shape):
            return (self.shape_in_play, self.curr_shape)
        
        return (new_shape, self.curr_shape)
    
    def spawn_shape(self):
        self.curr_shape = (self.WIDTH // 2, 0)
        self.shape_in_play = self.gen_new_shape()
    
    def is_dropped(self):
        return self.is_colliding(self.shape_in_play, (self.curr_shape[0], self.curr_shape[1] + 1))
    
    
    def get_clear_lines(self):
        clearable_lines = [np.all(self.board[:, i]) for i in range(self.board.shape[1])] # array dictating which lines are clearable
        new_board = np.zeros((self.HEIGHT, self.WIDTH))
        
        curr_line = self.board.shape[1] - 1 
        for i in [*range(self.board.shape[1])][::-1]:
            if not clearable_lines[i]:
                new_board[:, curr_line] = self.board[:, i]
                j -= 1
        self.score += get_score(clearable_lines.count(True))
        
        self.board = new_board
        
        return clearable_lines.count(True)
    
    
    def action_str_to_function(self):
        return {
            'shift_left': self.shift_left,
            'shift_right': self.shift_right,
            'rotate_counterclockwise': self.rotate_counterclockwise,
            'rotate_clockwise': self.rotate_clockwise,
            'soft_drop': self.soft_drop,
            'hard_drop': self.hard_drop,
            'none': lambda: (self.shape_in_play, self.curr_shape)
        }
    
    
    def get_no_valid_actions(self):
        no_valid_actions = 0
        
        for key in self.actions.keys():
            if self.action_str_to_function()[self.actions[key]]() == (self.shape_in_play, self.curr_shape):
                continue
            else:
                no_valid_actions += 1
        
        return no_valid_actions
    
    
    def clear_board(self):
        self.n_frames = 0
        self.score = 0
        
        self.gen_new_shape()
        
        self.board = np.zeros((self.HEIGHT, self.WIDTH))
        
        return self.board
    
    def _set_curr_piece(self, on_top=False):
        for (i, j) in self.shape_in_play:
            pixel_x, pixel_y = self.curr_shape[0] + i, self.curr_shape[1] + j
            if (pixel_x < self.WIDTH and pixel_x >= 0) and (pixel_y < self.HEIGHT and pixel_y >= 0):
                self.board[int(self.anchor[0] + i), int(self.anchor[1] + j)] = on_top
                
    def get_bumpiness(self):
        bumpiness = 0
        for i in range(self.board.shape[1] - 1):
            bumpiness += abs(self.board[:, i].sum() - self.board[:, i + 1].sum())
        return bumpiness
        
                
    def step(self, action):
        self.curr_shape = (int(self.curr_shape[0]), int(self.curr_shape[1]))
        
        self.shape_in_play, self.curr_shape = self.action_str_to_function()[self.actions[action]]()
        
        self.shape_in_play, self.curr_shape = self.soft_drop()
        
        self.n_frames += 1
        
        self._set_curr_piece()
        
        reward = self.get_no_valid_actions()
        
        done = False
        
        if self.is_dropped():
            self._set_curr_piece(on_top=True)
            reward += get_score(self.get_clear_lines())
            
            if np.any(self.board[0, :]):
                self.clear_board()
                
                self.att_num += 1
                
                done = True
                
                reward -= 300
            
            else:
                self.spawn_shape()
        
        self._set_curr_piece(on_top=True)
        state = deepcopy(self.board)
        self._set_curr_piece() 
        
        return state, reward, done
    
    def print_board(self):
        self._set_curr_piece(on_top=True)
        result_str = 'o' + '-' * self.WIDTH + 'o' + '\n'
        result_str += ('\n'.join(['|' + ''.join(['X' if j else ' ' for j in i]) + '|' for i in self.board.T]) + '\n')
        result_str += 'o' + '-' * self.WIDTH + 'o'
        print(result_str)
        
    
    def render(self):
        self._set_curr_piece(on_top=True)
        tmpboard = deepcopy(self.board)
        for i in tmpboard:
            for j in i:
                if j == 1:
                    j = 0
                else:
                    j = 255
        
        return cv2.resize(tmpboard, (self.WIDTH * 20, self.HEIGHT * 20), interpolation=cv2.INTER_NEAREST)
    
    def close():
        return
          