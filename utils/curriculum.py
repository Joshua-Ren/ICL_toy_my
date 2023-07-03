# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 22:52:05 2023

@author: YIREN
"""


class Curriculum:
    def __init__(self, args):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter
        self.curr_interval = args.curr_interval
        
        self.curr_dim_start = args.curr_dim_start
        self.curr_dim_end = args.n_dims
        self.curr_points_start = args.curr_points_start
        self.curr_points_end = args.prompt_points        
                
        self.n_dims = args.curr_dim_start
        self.n_points = args.curr_points_start
        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims = self.update_var(self.n_dims, self.curr_dim_end)
        self.n_points = self.update_var(self.n_points, self.curr_points_end)

    def update_var(self, var, end):
        if self.step_count % self.curr_interval == 0:
            var += 1
        return min(var, end)
