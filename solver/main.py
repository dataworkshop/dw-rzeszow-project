import logging
import sys

import falcon

sys.path.append('./')
import sudoku as s

logging.basicConfig(level=logging.INFO)


class TestResource(object):
    def on_get(self, req, res):
        sudoku_input = req.params['grid']
        logging.info('Solving sudoku for input params:' + sudoku_input)
        solution = self.solve(sudoku_input)
        logging.info('Solved!')
        res.status = falcon.HTTP_200
        res.body = str(solution)

    def reduce_puzzle(self, values, naked=True):
        """
        Iterate eliminate() and only_choice(). If at some point, there is a box with no available values, return False.
        If the sudoku is solved, return the sudoku.
        If after an iteration of both functions, the sudoku remains the same, return the sudoku.
        Input: A sudoku in dictionary form.
        Output: The resulting sudoku in dictionary form.
        """
        stalled = False
        while not stalled:
            solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
            values = s.eliminate(values)
            values = s.only_choice(values)
            if naked is True:
                values = s.naked_twins(values)
            solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
            stalled = solved_values_before == solved_values_after
            if len([box for box in values.keys() if len(values[box]) == 0]):
                return False
        return values

    def cross(self, a, b):
        return [s + t for s in a for t in b]

    def solve(self, diag_sudoku_grid):

        rows = 'ABCDEFGHI'
        cols = '123456789'

        boxes = self.cross(rows, cols)

        row_units = [self.cross(r, cols) for r in rows]
        column_units = [self.cross(rows, c) for c in cols]
        square_units = [self.cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]

        # Add diagonal units among the peers
        diagonal_units = [[r + c for r, c in zip(rows, cols)], [r + c for r, c in zip(rows[::-1], cols)]]
        unitlist = row_units + column_units + square_units + diagonal_units
        units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
        peers = dict((s, set(sum(units[s], [])) - set([s])) for s in boxes)

        grid_ = s.grid_values(diag_sudoku_grid)
        solved_puzzle = self.reduce_puzzle(grid_)
        return solved_puzzle


# Create the Falcon application object
app = falcon.API()

# Instantiate the TestResource class
test_resource = TestResource()

# Add a route to serve the resource
app.add_route('/sudoku', test_resource)
